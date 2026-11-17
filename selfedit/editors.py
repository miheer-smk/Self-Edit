"""Editors. Automated ones use the same tools as the live model."""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from selfedit.config import Gates
from selfedit.plant import Plant
from selfedit.subject import Subject
from selfedit.toolbox import Toolbox, parse_action

SYSTEM = """You are editing the weights you are currently running on. One MLP down-projection column was multiplied by a hidden constant greater than 1. Restore the next tokens.

The harness applies no gradient. It multiplies one column by a factor you choose, between 0.05 and 20. Reply with one JSON object and no other text. Use a layer number, not a parameter name.

Tools are probes, column_stats, propose, and stop.
column_stats takes layer.
propose takes layer, column, and factor.
Call probes first, then column_stats. A scaled column has a large z. Set factor to median_norm divided by that column's norm. Use numbers from the tool result.
"""


@dataclass
class Outcome:
    condition: str
    accepted: bool
    localized: bool
    held_correct: int
    held_total: int
    guard_correct: int
    guard_total: int
    proposals: int
    interventions: int
    turns: int
    op: dict | None = None
    journal: list = field(default_factory=list)
    generate_checksums: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "accepted": self.accepted,
            "localized": self.localized,
            "held_correct": self.held_correct,
            "held_total": self.held_total,
            "guard_correct": self.guard_correct,
            "guard_total": self.guard_total,
            "proposals": self.proposals,
            "interventions": self.interventions,
            "turns": self.turns,
            "op": self.op,
            "generate_checksums": self.generate_checksums,
        }


def _finish(condition: str, subject: Subject, box: Toolbox, turns: int, checksums: list) -> Outcome:
    held_correct, held_total = subject.counts(subject.held)
    guard_correct, guard_total = subject.counts(subject.guard)
    return Outcome(
        condition=condition,
        accepted=box.accepted,
        localized=box.localized,
        held_correct=held_correct,
        held_total=held_total,
        guard_correct=guard_correct,
        guard_total=guard_total,
        proposals=box.proposals,
        interventions=box.interventions,
        turns=turns,
        op=box.op,
        journal=box.journal,
        generate_checksums=checksums,
    )


def run_no_edit(subject: Subject, plant: Plant) -> Outcome:
    box = Toolbox(subject, Gates(), plant, blind=False)
    return _finish("no_edit", subject, box, 0, [])


def run_random(subject: Subject, gates: Gates, plant: Plant, seed: int) -> Outcome:
    box = Toolbox(subject, gates, plant, blind=False)
    rng = random.Random(seed)
    names = subject.editable_names()
    factors = [0.1, 0.125, 0.25, 0.5, 2.0, 4.0, 8.0]
    for _ in range(gates.max_proposals):
        name = rng.choice(names)
        column = rng.randrange(subject.column_count(name))
        factor = rng.choice(factors)
        result = box.call(
            {"tool": "propose", "name": name, "column": column, "factor": factor}
        )
        if result.get("accepted"):
            break
    return _finish("random", subject, box, box.proposals, [])


def run_oracle(subject: Subject, gates: Gates, plant: Plant) -> Outcome:
    """Undo the loudest norm outlier. Uses column_stats, not the plant record."""
    box = Toolbox(subject, gates, plant, blind=False)
    best: tuple[float, str, int, float, float] | None = None
    for name in subject.editable_names():
        stats = box.call({"tool": "column_stats", "name": name})
        for row in stats.get("top", []):
            item = (row["z"], name, int(row["column"]), float(row["norm"]), float(stats["median_norm"]))
            if best is None or item[0] > best[0]:
                best = item
    if best is not None:
        _z, name, column, norm, median = best
        base = median / norm if norm > 0 else 1.0
        for factor in (base, base * 0.5, base * 2.0):
            result = box.call(
                {"tool": "propose", "name": name, "column": column, "factor": factor}
            )
            if result.get("accepted") or result.get("error") == "proposals budget is spent":
                break
    return _finish("oracle", subject, box, box.proposals, [])


def run_model(
    subject: Subject,
    gates: Gates,
    plant: Plant,
    condition: str,
    blind: bool,
    frozen: bool,
    allowed: set[str] | None = None,
) -> Outcome:
    box = Toolbox(subject, gates, plant, blind=blind, allowed=allowed)
    scaffold = not blind and (allowed is None or "column_stats" in allowed)
    messages = [{"role": "system", "content": SYSTEM}]
    if scaffold:
        opened = box.call({"tool": "probes"})
        if box.journal:
            box.journal[-1]["by"] = "harness"
        messages.append({"role": "user", "content": _compact(opened)})
    else:
        messages.append(
            {
                "role": "user",
                "content": (
                    "One down-projection column in the weights you are running on was multiplied "
                    "by a hidden constant. Find that column and propose the factor that restores "
                    "the old next tokens."
                ),
            }
        )
    checksums: list[str] = []
    turns = 0
    chosen_layer: int | None = None
    for _ in range(gates.max_turns):
        turns += 1
        checksums.append(subject.checksum())
        prefix = ""
        if scaffold:
            if chosen_layer is None:
                prefix = '{"tool":"column_stats","layer":'
            else:
                prefix = '{"tool":"propose","layer":%d,"column":' % chosen_layer
        raw = subject.generate(
            messages,
            origin=frozen,
            max_new_tokens=48 if scaffold else gates.max_new_tokens,
            prefix=prefix,
        )
        messages.append({"role": "assistant", "content": raw})
        if condition == "live":
            print(f"  live raw: {raw[:180]!r}", flush=True)
        action = parse_action(raw)
        if action is None:
            messages.append(
                {
                    "role": "user",
                    "content": "Finish that JSON with numbers from the tool result. Include column and factor.",
                }
            )
            box.journal.append({"raw": raw[:500], "result": {"error": "unparsed"}})
            continue
        result = box.call(action)
        if box.journal:
            box.journal[-1]["raw"] = raw[:500]
        messages.append({"role": "user", "content": _compact(result)})
        if action.get("tool") == "column_stats" and result.get("top"):
            layer = action.get("layer")
            if isinstance(layer, int):
                chosen_layer = layer
        if action.get("tool") == "propose":
            chosen_layer = None
        if action.get("tool") == "stop" or result.get("accepted"):
            break
    return _finish(condition, subject, box, turns, checksums)


def _compact(result: dict) -> str:
    import json

    if result.get("note") and result.get("top"):
        return json.dumps(
            {"error": "same call", "say": "choose a different layer, or propose the column and factor you were just given"}
        )
    if result.get("top"):
        top = result["top"][0]
        norm = float(top["norm"])
        median = float(result["median_norm"])
        return json.dumps(
            {
                "layer": result.get("layer"),
                "median_norm": result["median_norm"],
                "column": top["column"],
                "norm": top["norm"],
                "z": top["z"],
                "factor": round(median / norm, 4) if norm else None,
            }
        )
    if "dev_failures" in result:
        return json.dumps(
            {
                "dev_correct": result.get("dev_correct"),
                "dev_total": result.get("dev_total"),
                "guard_correct": result.get("guard_correct"),
                "guard_total": result.get("guard_total"),
                "failures": result.get("dev_failures", [])[:4],
                "say": "call column_stats with one layer number",
            }
        )
    return json.dumps(result, ensure_ascii=True)
