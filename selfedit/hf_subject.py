"""Hugging Face causal LM as the live subject and the editor."""

from __future__ import annotations

import torch

from selfedit.probes import PROMPTS
from selfedit.subject import Subject


class HFSubject(Subject):
    def __init__(self, model_id: str, device: str = "auto") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if device == "auto":
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = torch.device(device)
        dtype = torch.float16 if self.device.type != "cpu" else torch.float32
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        gen = self.model.generation_config
        gen.do_sample = False
        gen.temperature = None
        gen.top_p = None
        gen.top_k = None
        self.prompts = list(PROMPTS)
        self.gold = []
        self.dev = []
        self.held = []
        self.guard = []
        self.origin: dict[str, torch.Tensor] | None = None
        self.commit = getattr(self.model.config, "_commit_hash", None)
        self.dtype = str(dtype).replace("torch.", "")
        n_params = sum(p.numel() for p in self.model.parameters())
        print(
            f"loaded {model_id} params={n_params} device={self.device} dtype={self.dtype}",
            flush=True,
        )

    def named_params(self):
        return list(self.model.named_parameters())

    def label(self, token_id: int) -> str:
        return self.tokenizer.decode([token_id])

    def _encoded(self):
        encoded = self.tokenizer(self.prompts, return_tensors="pt", padding=True)
        return {key: value.to(self.device) for key, value in encoded.items()}

    def predict_map(self) -> list[int]:
        encoded = self._encoded()
        with torch.inference_mode():
            logits = self.model(**encoded).logits
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        index = torch.arange(lengths.shape[0], device=lengths.device)
        return logits[index, lengths].argmax(dim=-1).tolist()

    def activation_matrix(self, name: str) -> torch.Tensor:
        module_name = name.rsplit(".", 1)[0]
        modules = dict(self.model.named_modules())
        if module_name not in modules:
            raise KeyError(name)
        encoded = self._encoded()
        lengths = encoded["attention_mask"].sum(dim=1) - 1
        holder: dict[str, torch.Tensor] = {}

        def pre_hook(module, args):
            hidden = args[0]
            batch = torch.arange(hidden.shape[0], device=hidden.device)
            holder["a"] = hidden[batch, lengths.to(hidden.device)].detach().float().cpu()

        handle = modules[module_name].register_forward_pre_hook(pre_hook)
        try:
            with torch.inference_mode():
                self.model(**encoded)
        finally:
            handle.remove()
        if "a" not in holder:
            raise RuntimeError(f"no activation captured for {name}")
        return holder["a"].abs()

    def generate(
        self,
        messages: list[dict],
        origin: bool = False,
        max_new_tokens: int = 256,
        prefix: str = "",
    ) -> str:
        scratch = self.clone_state() if origin else None
        if origin:
            if self.origin is None:
                raise RuntimeError("original checkpoint was not saved")
            self.load_state(self.origin)
        try:
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if not isinstance(text, str):
                raise RuntimeError("chat template did not return a string")
            text += prefix
            encoded = self.tokenizer(text, return_tensors="pt")
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                output = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            new_tokens = output[0, encoded["input_ids"].shape[1] :]
            continuation = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            if prefix and continuation.startswith(prefix):
                return continuation
            return prefix + continuation
        finally:
            if scratch is not None:
                self.load_state(scratch)
