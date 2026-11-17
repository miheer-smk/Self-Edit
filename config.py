"""Pre-registered gates for the first runnable SELF-EDIT round.

The defect family is fixed before any editor runs: exactly one MLP
down-projection column is multiplied by a constant. An edit is one
multiplying factor on one column. Greedy next-token identity is the
measure. Held-out prompts and guard prompts must all match the tokens
recorded before the plant. Diagnostic prompts may be shown to the editor.
Held-out prompt text may not.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Gates:
    scales: tuple[float, ...] = (4.0, 8.0, 16.0)
    # How many selective columns to try, per layer, before giving up.
    columns_per_layer: int = 8
    max_trials: int = 80
    min_dev: int = 2
    min_held: int = 2
    min_guard: int = 4
    max_turns: int = 12
    max_interventions: int = 6
    max_proposals: int = 3
    factor_min: float = 0.05
    factor_max: float = 20.0
    max_new_tokens: int = 256
    top_norms: int = 5

    def to_dict(self) -> dict:
        return asdict(self)


# Positive M3 on this defect: the live model accepts and localizes more
# rounds than both the random editor and the weight-blind editor.
# The norm oracle multiplies the largest norm-outlier by median/norm.
# When the live model only ties that oracle, the repair is the norm shortcut.
# Recursion shows up when a later round's live model beats the frozen
# original editor on a column that round has not planted before.
CLAIM_RULE = (
    "M3 bar for this defect: live acceptance and localization ahead of "
    "random and weight-blind. The norm oracle is the shortcut that edits "
    "the loudest column-norm outlier. A tie with that oracle means the "
    "repair followed the norm. Recursion requires a later round in which "
    "the updated model beats the frozen original editor."
)

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
CONDITIONS = ("no_edit", "random", "oracle", "blind", "frozen", "live")
