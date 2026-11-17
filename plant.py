"""Search for one scaled column that moves some prompts and leaves others."""

from __future__ import annotations

from dataclasses import dataclass

from selfedit.config import Gates
from selfedit.subject import Subject


@dataclass
class Plant:
    name: str
    column: int
    scale: float
    trials: int
    dev: list[int]
    held: list[int]
    guard: list[int]

    def to_dict(self, prompts: list[str]) -> dict:
        return {
            "name": self.name,
            "column": self.column,
            "scale": self.scale,
            "trials": self.trials,
            "dev_prompts": [prompts[i] for i in self.dev],
            "held_prompts": [prompts[i] for i in self.held],
            "guard_prompts": [prompts[i] for i in self.guard],
        }


def _split(flipped: list[int]) -> tuple[list[int], list[int]]:
    return flipped[0::2], flipped[1::2]


def _ranked_columns(subject: Subject, name: str, keep: int) -> list[int]:
    matrix = subject.activation_matrix(name)
    score = matrix.std(dim=0, unbiased=False)
    order = torch_argsort_desc(score)
    return [int(i) for i in order[:keep]]


def torch_argsort_desc(score) -> list[int]:
    return score.argsort(descending=True).tolist()


def find_plant(
    subject: Subject,
    gates: Gates,
    banned: set[tuple[str, int]],
) -> Plant | None:
    """Leave a valid plant applied, or restore the starting weights."""
    start = subject.clone_state()
    subject.gold = subject.predict_map()
    names = subject.editable_names()
    if not names:
        subject.load_state(start)
        return None
    ranked = {
        name: _ranked_columns(subject, name, gates.columns_per_layer) for name in names
    }
    trials = 0
    for scale in gates.scales:
        for rank in range(gates.columns_per_layer):
            for name in names:
                if rank >= len(ranked[name]):
                    continue
                column = ranked[name][rank]
                if (name, column) in banned:
                    continue
                trials += 1
                subject.scale_column(name, column, scale)
                preds = subject.predict_map()
                flipped = [i for i, pred in enumerate(preds) if pred != subject.gold[i]]
                stable = [i for i, pred in enumerate(preds) if pred == subject.gold[i]]
                dev, held = _split(flipped)
                ok = (
                    len(dev) >= gates.min_dev
                    and len(held) >= gates.min_held
                    and len(stable) >= gates.min_guard
                )
                print(
                    f"plant trial {trials}: {name} col {column} x{scale:g} "
                    f"flipped={len(flipped)} stable={len(stable)}",
                    flush=True,
                )
                if ok:
                    subject.dev, subject.held, subject.guard = dev, held, stable
                    return Plant(name, column, scale, trials, dev, held, stable)
                subject.load_state(start)
                if trials >= gates.max_trials:
                    subject.dev, subject.held, subject.guard = [], [], []
                    return None
    subject.load_state(start)
    subject.dev, subject.held, subject.guard = [], [], []
    return None
