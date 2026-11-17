"""Live parameters, snapshots, and next-token scores."""

from __future__ import annotations

import hashlib
from typing import Iterable

import torch
import torch.nn as nn


def _editable(name: str, param: torch.Tensor) -> bool:
    if param.ndim != 2:
        return False
    return name == "down_proj" or name.endswith("mlp.down_proj.weight")


class Subject:
    """A checkpoint the editor and the harness share.

    Columns are dimension 1 of an editable matrix: one intermediate neuron
    of an MLP down-projection, or one input axis of the tiny stand-in.
    """

    prompts: list[str]
    gold: list[int]
    dev: list[int]
    held: list[int]
    guard: list[int]

    def named_params(self) -> list[tuple[str, torch.nn.Parameter]]:
        raise NotImplementedError

    def predict_map(self) -> list[int]:
        raise NotImplementedError

    def activation_matrix(self, name: str) -> torch.Tensor:
        """Absolute activations, shape [n_prompts, n_columns]."""
        raise NotImplementedError

    def label(self, token_id: int) -> str:
        raise NotImplementedError

    def generate(
        self,
        messages: list[dict],
        origin: bool = False,
        max_new_tokens: int = 256,
        prefix: str = "",
    ) -> str:
        raise NotImplementedError

    def editable_names(self) -> list[str]:
        return [name for name, param in self.named_params() if _editable(name, param)]

    def param(self, name: str) -> torch.nn.Parameter:
        found = dict(self.named_params())
        if name not in found:
            raise KeyError(name)
        return found[name]

    def column_count(self, name: str) -> int:
        return int(self.param(name).shape[1])

    def column_norms(self, name: str) -> torch.Tensor:
        weight = self.param(name).detach().float().cpu()
        return weight.norm(dim=0)

    def scale_column(self, name: str, column: int, factor: float) -> None:
        weight = self.param(name)
        if column < 0 or column >= weight.shape[1]:
            raise IndexError(column)
        with torch.no_grad():
            weight.data[:, column].mul_(float(factor))

    def column_summary(self, name: str, column: int) -> dict:
        weight = self.param(name).detach().float().cpu()
        vector = weight[:, column]
        mean_column = weight.mean(dim=1)
        denom = vector.norm() * mean_column.norm()
        cosine = float(torch.dot(vector, mean_column) / denom) if denom > 0 else 0.0
        return {
            "column": column,
            "norm": float(vector.norm()),
            "mean": float(vector.mean()),
            "std": float(vector.std(unbiased=False)),
            "cosine_to_mean_column": cosine,
        }

    def counts(self, indices: Iterable[int]) -> tuple[int, int]:
        indices = list(indices)
        if not indices:
            return 0, 0
        preds = self.predict_map()
        correct = sum(preds[i] == self.gold[i] for i in indices)
        return correct, len(indices)

    def clone_state(self) -> dict[str, torch.Tensor]:
        return {name: param.detach().cpu().clone() for name, param in self.named_params()}

    def load_state(self, state: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for name, param in self.named_params():
                param.copy_(state[name].to(device=param.device, dtype=param.dtype))

    def checksum(self) -> str:
        digest = hashlib.sha256()
        for name in self.editable_names():
            digest.update(name.encode())
            digest.update(self.column_norms(name).numpy().tobytes())
        return digest.hexdigest()[:16]


class _TinyNet(nn.Module):
    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        self.down_proj = nn.Parameter(weight, requires_grad=False)


class TinySubject(Subject):
    """Small linear map used to test the harness without a download.

    Column 0 is a shared rival direction, loaded weakly by every signal
    prompt. Each signal prompt also has its own stronger private direction.
    Multiplying column 0 by 4 makes the rival win on every signal prompt
    and leaves the steady prompts alone. Every column starts at norm 1,
    so the planted column is the norm outlier.
    """

    def __init__(self, seed: int = 0) -> None:
        del seed  # The map is fixed so the plant search has a known answer.
        n_out = 8
        n_signal = 16
        n_steady = 16
        n_in = 1 + n_signal + n_steady
        weight = torch.zeros(n_out, n_in)
        weight[0, 0] = 1.0
        self.X = torch.zeros(n_signal + n_steady, n_in)
        self.prompts = []
        for i in range(n_signal):
            target = 1 + (i % (n_out - 1))
            weight[target, 1 + i] = 1.0
            self.X[i, 0] = 0.5
            self.X[i, 1 + i] = 1.0
            self.prompts.append(f"signal-{i:02d}")
        for j in range(n_steady):
            weight[j % n_out, 1 + n_signal + j] = 1.0
            self.X[n_signal + j, 1 + n_signal + j] = 1.0
            self.prompts.append(f"steady-{j:02d}")
        self.net = _TinyNet(weight)
        self.gold = []
        self.dev = []
        self.held = []
        self.guard = []
        self.origin: dict[str, torch.Tensor] | None = None

    def named_params(self) -> list[tuple[str, torch.nn.Parameter]]:
        return list(self.net.named_parameters())

    def predict_map(self) -> list[int]:
        with torch.no_grad():
            logits = self.X @ self.param("down_proj").detach().float().cpu().T
        return logits.argmax(dim=-1).tolist()

    def activation_matrix(self, name: str) -> torch.Tensor:
        if name != "down_proj":
            raise KeyError(name)
        return self.X.abs()

    def label(self, token_id: int) -> str:
        return str(token_id)

    def generate(
        self,
        messages: list[dict],
        origin: bool = False,
        max_new_tokens: int = 256,
        prefix: str = "",
    ) -> str:
        raise RuntimeError("the tiny subject has no language editor")
