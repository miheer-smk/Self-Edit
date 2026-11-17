"""Tools the editor may call. The harness applies them and scores them."""

from __future__ import annotations

import json
import re

from selfedit.config import Gates
from selfedit.plant import Plant
from selfedit.subject import Subject

WEIGHT_TOOLS = {"column_stats", "column", "activations", "intervene"}


class Toolbox:
    def __init__(
        self,
        subject: Subject,
        gates: Gates,
        plant: Plant,
        blind: bool,
        allowed: set[str] | None = None,
    ) -> None:
        self.subject = subject
        self.gates = gates
        self.plant = plant
        self.blind = blind
        self.allowed = allowed
        self.interventions = 0
        self.proposals = 0
        self.accepted = False
        self.localized = False
        self.op: dict | None = None
        self.last_scores: dict = {}
        self.journal: list[dict] = []

    def call(self, action: dict) -> dict:
        tool = action.get("tool")
        previous = self.journal[-1].get("action") if self.journal else None
        repeated = bool(previous) and previous.get("tool") == tool
        if self.allowed is not None and tool not in self.allowed:
            result = {"error": "this tool is outside the current information rung"}
        elif self.blind and tool in WEIGHT_TOOLS:
            result = {"error": "weight and activation tools are off in this condition"}
        else:
            result = self._dispatch(action)
        if repeated and isinstance(result, dict) and "error" not in result:
            result["note"] = "You already received this. Choose a different tool."
        # Operator log. This object is not returned to the model.
        self.journal.append({"action": action, "result": result, "localized": self.localized})
        return result

    def _dispatch(self, action: dict) -> dict:
        tool = action.get("tool")
        if tool == "manifest":
            editable = []
            others = 0
            for name, param in self.subject.named_params():
                row = {"name": name, "shape": list(param.shape)}
                if name in self.subject.editable_names():
                    editable.append(row)
                else:
                    others += 1
            if action.get("scope") == "all":
                return {
                    "parameters": editable
                    + [
                        {"name": name, "shape": list(param.shape)}
                        for name, param in self.subject.named_params()
                        if name not in self.subject.editable_names()
                    ],
                    "editable": [row["name"] for row in editable],
                }
            return {
                "editable": editable,
                "n_other_parameters": others,
                "patch_scope": "mlp down-projection columns only",
            }
        if tool == "probes":
            return self._probes()
        if tool == "column_stats":
            return self._column_stats(action)
        if tool == "column":
            return self._column(action)
        if tool == "activations":
            return self._activations(action)
        if tool == "intervene":
            return self._scale(action, keep=False)
        if tool == "propose":
            return self._scale(action, keep=True)
        if tool == "stop":
            return {"stopped": True}
        return {"error": "unknown tool"}

    def _probes(self) -> dict:
        preds = self.subject.predict_map()
        failures = []
        for index in self.subject.dev:
            now = preds[index]
            if now == self.subject.gold[index]:
                continue
            failures.append(
                {
                    "prompt": self.subject.prompts[index],
                    "before": self.subject.label(self.subject.gold[index]),
                    "now": self.subject.label(now),
                }
            )
        dev_correct, dev_n = self.subject.counts(self.subject.dev)
        guard_correct, guard_n = self.subject.counts(self.subject.guard)
        return {
            "dev_failures": failures,
            "dev_correct": dev_correct,
            "dev_total": dev_n,
            "guard_correct": guard_correct,
            "guard_total": guard_n,
            "heldout_hidden": len(self.subject.held),
            "layers": _layers(self.subject),
            "next": "Call column_stats with a layer number. Then propose that layer, the top column, and factor = median_norm / norm.",
        }

    def _require_editable(self, action: dict) -> str | dict:
        layer = _as_int(action.get("layer"))
        if layer is not None:
            matches = [name for name in self.subject.editable_names() if f".layers.{layer}." in name]
            if len(matches) == 1:
                return matches[0]
            return {"error": "layer is not an editable down-projection", "layers": _layers(self.subject)}
        name = action.get("name")
        if not isinstance(name, str) or name not in self.subject.editable_names():
            return {
                "error": "use a layer number",
                "layers": _layers(self.subject),
            }
        return name

    def _column_stats(self, action: dict) -> dict:
        name = self._require_editable(action)
        if isinstance(name, dict):
            return name
        norms = self.subject.column_norms(name)
        mean = float(norms.mean())
        std = float(norms.std(unbiased=False))
        median = float(norms.median())
        z = (norms - mean) / std if std > 0 else norms * 0
        top_index = z.argsort(descending=True)[: self.gates.top_norms].tolist()
        top = []
        for column in top_index:
            top.append(
                {
                    "column": int(column),
                    "norm": round(float(norms[column]), 4),
                    "z": round(float(z[column]), 4),
                }
            )
        layer = _layer_of(name)
        return {
            "layer": layer,
            "name": name,
            "n_columns": int(norms.numel()),
            "median_norm": round(median, 4),
            "mean_norm": round(mean, 4),
            "std_norm": round(std, 4),
            "top": top,
        }

    def _column(self, action: dict) -> dict:
        name = self._require_editable(action)
        if isinstance(name, dict):
            return name
        column = _as_int(action.get("column"))
        if column is None:
            return {"error": "column must be an integer"}
        try:
            summary = self.subject.column_summary(name, column)
        except IndexError:
            return {"error": "column is out of range"}
        summary["name"] = name
        return summary

    def _activations(self, action: dict) -> dict:
        name = self._require_editable(action)
        if isinstance(name, dict):
            return name
        prompt = action.get("prompt")
        index = None
        for dev_index in self.subject.dev:
            if self.subject.prompts[dev_index] == prompt:
                index = dev_index
                break
        if index is None:
            return {"error": "prompt is not in the diagnostic set"}
        matrix = self.subject.activation_matrix(name)
        row = matrix[index]
        top_index = row.argsort(descending=True)[: self.gates.top_norms].tolist()
        return {
            "name": name,
            "prompt": prompt,
            "top": [{"column": int(i), "activation": float(row[i])} for i in top_index],
        }

    def _scale(self, action: dict, keep: bool) -> dict:
        name = self._require_editable(action)
        if isinstance(name, dict):
            return name
        column = _as_int(action.get("column"))
        factor = _as_float(action.get("factor"))
        if column is None or factor is None:
            return {"error": "column and factor are required"}
        if not (self.gates.factor_min <= factor <= self.gates.factor_max):
            return {"error": "factor is outside the pre-registered range"}
        budget = self.proposals if keep else self.interventions
        limit = self.gates.max_proposals if keep else self.gates.max_interventions
        if budget >= limit:
            which = "proposals" if keep else "interventions"
            return {"error": f"{which} budget is spent"}
        try:
            saved = self.subject.param(name).data[:, column].detach().clone()
        except IndexError:
            return {"error": "column is out of range"}
        self.subject.scale_column(name, column, factor)
        if keep:
            self.proposals += 1
            held_correct, held_n = self.subject.counts(self.subject.held)
            guard_correct, guard_n = self.subject.counts(self.subject.guard)
            passed = held_correct == held_n and guard_correct == guard_n and held_n > 0
            scores = {
                "accepted": passed,
                "held_correct": held_correct,
                "held_total": held_n,
                "guard_correct": guard_correct,
                "guard_total": guard_n,
            }
            if passed:
                self.accepted = True
                self.localized = name == self.plant.name and column == self.plant.column
                self.op = {"name": name, "column": column, "factor": factor}
                self.last_scores = scores
                return scores
            _restore_column(self.subject, name, column, saved)
            self.last_scores = scores
            return scores
        self.interventions += 1
        dev_correct, dev_n = self.subject.counts(self.subject.dev)
        guard_correct, guard_n = self.subject.counts(self.subject.guard)
        _restore_column(self.subject, name, column, saved)
        return {
            "accepted": False,
            "reverted": True,
            "dev_correct": dev_correct,
            "dev_total": dev_n,
            "guard_correct": guard_correct,
            "guard_total": guard_n,
        }


def _restore_column(subject: Subject, name: str, column: int, saved) -> None:
    import torch

    weight = subject.param(name)
    with torch.no_grad():
        weight.data[:, column].copy_(saved.to(device=weight.device, dtype=weight.dtype))


def _as_int(value) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _as_float(value) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def parse_action(text: str) -> dict | None:
    for line in text.splitlines():
        candidate = line.strip().strip("`")
        if candidate.startswith("{") and candidate.endswith("}"):
            action = _json_tool(candidate)
            if action is not None:
                return action
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        action = _json_tool(text[start : end + 1])
        if action is not None:
            return action
    for line in text.splitlines():
        exact = line.strip().strip("`")
        stats = re.fullmatch(r"column_stats\s+(\d+)", exact, re.IGNORECASE)
        if stats:
            return {"tool": "column_stats", "layer": int(stats.group(1))}
        propose = re.fullmatch(
            r"propose\s+(\d+)\s+(\d+)\s+([0-9]*\.?[0-9]+)",
            exact,
            re.IGNORECASE,
        )
        if propose:
            return {
                "tool": "propose",
                "layer": int(propose.group(1)),
                "column": int(propose.group(2)),
                "factor": float(propose.group(3)),
            }
        if re.fullmatch(r"probes", exact, re.IGNORECASE):
            return {"tool": "probes"}
        if re.fullmatch(r"stop", exact, re.IGNORECASE):
            return {"tool": "stop"}
    stats = re.search(
        r'"tool"\s*:\s*"column_stats"\s*,\s*"layer"\s*:\s*(\d+)',
        text,
    )
    if stats:
        return {"tool": "column_stats", "layer": int(stats.group(1))}
    propose = re.search(
        r'"tool"\s*:\s*"propose"\s*,\s*"layer"\s*:\s*(\d+)\s*,\s*"column"\s*:\s*(\d+)\s*,\s*"factor"\s*:\s*([0-9]*\.?[0-9]+)',
        text,
    )
    if propose:
        return {
            "tool": "propose",
            "layer": int(propose.group(1)),
            "column": int(propose.group(2)),
            "factor": float(propose.group(3)),
        }
    return None


def _json_tool(text: str) -> dict | None:
    try:
        action = json.loads(text)
    except json.JSONDecodeError:
        return None
    if isinstance(action, dict) and isinstance(action.get("tool"), str):
        return action
    return None


def _layers(subject: Subject) -> list[int]:
    return [layer for name in subject.editable_names() if (layer := _layer_of(name)) is not None]


def _layer_of(name: str) -> int | None:
    match = re.search(r"layers\.(\d+)\.", name)
    return int(match.group(1)) if match else None
