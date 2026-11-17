"""Run the harness self-test or a real M3 round."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

from selfedit.config import CLAIM_RULE, CONDITIONS, DEFAULT_MODEL, Gates
from selfedit.editors import run_model, run_no_edit, run_oracle, run_random
from selfedit.plant import Plant, find_plant
from selfedit.subject import Subject, TinySubject

ROOT = Path(__file__).resolve().parents[1]


def run_selftest(out_dir: Path | None = None) -> dict:
    torch.set_grad_enabled(False)
    gates = Gates()
    subject = TinySubject(seed=0)
    subject.origin = subject.clone_state()

    first = subject.checksum()
    subject.predict_map()
    if subject.checksum() != first:
        raise AssertionError("M0 failed: a forward pass changed the weights")

    subject.scale_column("down_proj", 0, 8.0)
    if subject.checksum() == first:
        raise AssertionError("M2 failed: scaling a column left the checksum unchanged")
    subject.load_state(subject.origin)
    if subject.checksum() != first:
        raise AssertionError("M2 failed: rollback did not restore the weights")

    plant = find_plant(subject, gates, banned=set())
    if plant is None:
        raise AssertionError("plant search found no selective column on the tiny map")
    planted = subject.clone_state()
    planted_checksum = subject.checksum()

    _assert_heldout_hidden(subject, gates, plant)

    before = subject.checksum()
    box_probe = _toolbox(subject, gates, plant)
    box_probe.call(
        {"tool": "intervene", "name": plant.name, "column": plant.column, "factor": 0.5}
    )
    if subject.checksum() != before:
        raise AssertionError("intervene did not restore the column")

    subject.load_state(planted)
    oracle = run_oracle(subject, gates, plant)
    if not oracle.accepted or not oracle.localized:
        raise AssertionError(
            f"norm oracle failed to undo the planted column: {oracle.to_dict()}"
        )
    if subject.checksum() == planted_checksum:
        raise AssertionError("accepted oracle edit did not change the planted weights")

    subject.load_state(planted)
    no_edit = run_no_edit(subject, plant)
    if no_edit.accepted:
        raise AssertionError("no-edit passed the gate on a planted defect")

    wins = 0
    for seed in range(40):
        subject.load_state(planted)
        if run_random(subject, gates, plant, seed).accepted:
            wins += 1
    if wins > 6:
        raise AssertionError(f"random editor accepted {wins}/40, above the 6/40 cap")

    subject.load_state(subject.origin)
    report = {
        "stage": "selftest",
        "claim_rule": CLAIM_RULE,
        "gates": gates.to_dict(),
        "plant": plant.to_dict(subject.prompts),
        "oracle": oracle.to_dict(),
        "no_edit_accepted": no_edit.accepted,
        "random_accepted_out_of_40": wins,
        "rollback_checksum": subject.checksum(),
    }
    destination = out_dir or (ROOT / "runs" / "selftest")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"selftest passed: oracle localized the x{plant.scale:g} column, "
        f"random accepted {wins}/40, rollback restored the checksum",
        flush=True,
    )
    print(f"wrote {destination / 'report.json'}", flush=True)
    return report


def run_m3(
    model_id: str = DEFAULT_MODEL,
    rounds: int = 3,
    seed: int = 0,
    conditions: tuple[str, ...] = CONDITIONS,
    device: str = "auto",
    out_dir: Path | None = None,
) -> dict:
    torch.set_grad_enabled(False)
    from selfedit.hf_subject import HFSubject

    unknown = [name for name in conditions if name not in CONDITIONS]
    if unknown:
        raise SystemExit(f"unknown condition: {', '.join(unknown)}")
    gates = Gates()
    subject = HFSubject(model_id, device=device)
    if not subject.editable_names():
        raise SystemExit("this checkpoint has no mlp.down_proj.weight tensors")
    subject.origin = subject.clone_state()
    banned: set[tuple[str, int]] = set()
    round_reports = []
    journals = []
    for round_index in range(rounds):
        print(f"round {round_index + 1}/{rounds}", flush=True)
        before = subject.clone_state()
        clean_checksum = subject.checksum()
        plant = find_plant(subject, gates, banned)
        if plant is None:
            round_reports.append(
                {
                    "round": round_index,
                    "status": "plant_failed",
                    "clean_checksum": clean_checksum,
                }
            )
            continue
        banned.add((plant.name, plant.column))
        planted_state = subject.clone_state()
        planted_checksum = subject.checksum()
        condition_reports = {}
        for name in conditions:
            subject.load_state(planted_state)
            outcome = _run_condition(name, subject, gates, plant, seed + 1000 * round_index)
            condition_reports[name] = outcome.to_dict()
            for entry in outcome.journal:
                journals.append({"round": round_index, "condition": name, **entry})
            print(
                f"  {name}: accepted={outcome.accepted} localized={outcome.localized} "
                f"held={outcome.held_correct}/{outcome.held_total} "
                f"guard={outcome.guard_correct}/{outcome.guard_total}",
                flush=True,
            )
        if "live" in condition_reports and condition_reports["live"]["accepted"]:
            subject.load_state(planted_state)
            op = condition_reports["live"]["op"]
            subject.scale_column(op["name"], op["column"], op["factor"])
            status = "committed"
        else:
            subject.load_state(before)
            status = "rolled_back"
        round_reports.append(
            {
                "round": round_index,
                "status": status,
                "clean_checksum": clean_checksum,
                "planted_checksum": planted_checksum,
                "plant": plant.to_dict(subject.prompts),
                "conditions": condition_reports,
            }
        )
    summary = _summarize(round_reports)
    report = {
        "stage": "M3",
        "model": model_id,
        "commit": subject.commit,
        "device": str(subject.device),
        "dtype": subject.dtype,
        "gates": gates.to_dict(),
        "claim_rule": CLAIM_RULE,
        "conditions": list(conditions),
        "rounds": round_reports,
        "summary": summary,
        "decision": _decision(summary),
    }
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = out_dir or (ROOT / "runs" / f"m3-{stamp}")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (destination / "journal.jsonl").write_text(
        "".join(json.dumps(entry, ensure_ascii=True) + "\n" for entry in journals)
    )
    print(_decision(summary), flush=True)
    print(f"wrote {destination / 'report.json'}", flush=True)
    return report


def _run_condition(name: str, subject: Subject, gates: Gates, plant: Plant, seed: int):
    if name == "no_edit":
        return run_no_edit(subject, plant)
    if name == "random":
        return run_random(subject, gates, plant, seed)
    if name == "oracle":
        return run_oracle(subject, gates, plant)
    if name == "blind":
        return run_model(subject, gates, plant, "blind", blind=True, frozen=False)
    if name == "frozen":
        return run_model(subject, gates, plant, "frozen", blind=False, frozen=True)
    if name == "live":
        return run_model(subject, gates, plant, "live", blind=False, frozen=False)
    raise KeyError(name)


def _toolbox(subject: Subject, gates: Gates, plant: Plant):
    from selfedit.toolbox import Toolbox

    return Toolbox(subject, gates, plant, blind=False)


def _assert_heldout_hidden(subject: Subject, gates: Gates, plant: Plant) -> None:
    box = _toolbox(subject, gates, plant)
    probes = box.call({"tool": "probes"})
    blob = json.dumps(probes)
    if "localized" in probes or "secret" in blob:
        raise AssertionError("probe payload exposes the plant")
    for index in plant.held:
        if subject.prompts[index] in blob:
            raise AssertionError("probe payload includes a held-out prompt")


def _summarize(rounds: list[dict]) -> dict:
    planted = [row for row in rounds if row.get("plant")]
    summary = {
        "rounds": len(rounds),
        "planted_rounds": len(planted),
        "plant_failed": sum(row.get("status") == "plant_failed" for row in rounds),
    }
    for name in CONDITIONS:
        present = [row["conditions"][name] for row in planted if name in row.get("conditions", {})]
        summary[f"{name}_accepted"] = sum(item["accepted"] for item in present)
        summary[f"{name}_localized"] = sum(item["localized"] for item in present)
        summary[f"{name}_rounds"] = len(present)
    later = [row for row in planted if row["round"] > 0 and "live" in row["conditions"]]
    summary["live_later_accepted"] = sum(row["conditions"]["live"]["accepted"] for row in later)
    summary["frozen_later_accepted"] = sum(
        row["conditions"]["frozen"]["accepted"] for row in later if "frozen" in row["conditions"]
    )
    summary["later_rounds"] = len(later)
    return summary


def _decision(summary: dict) -> str:
    if summary["planted_rounds"] == 0:
        return "No selective column was found, so no editor was scored."
    labels = (
        ("live", "Live"),
        ("random", "Random"),
        ("oracle", "The norm oracle"),
        ("blind", "Weight-blind"),
        ("frozen", "The frozen editor"),
        ("no_edit", "No-edit"),
    )
    parts = []
    for name, label in labels:
        n = summary.get(f"{name}_rounds", 0)
        if not n:
            continue
        localized = ""
        if name == "live":
            localized = f" and localized {summary['live_localized']}"
        parts.append(f"{label} accepted {summary[f'{name}_accepted']} of {n}{localized}")
    lines = [". ".join(parts) + "."]
    if summary.get("live_rounds", 0) and summary.get("random_rounds", 0) and summary.get("blind_rounds", 0):
        live = summary["live_accepted"]
        ahead = (
            live > summary["random_accepted"]
            and live > summary["blind_accepted"]
            and summary["live_localized"] > 0
        )
        if ahead:
            lines.append("This run meets the M3 bar for a scaled-column defect.")
        else:
            lines.append("This run is below the M3 bar for a scaled-column defect.")
    if (
        summary.get("live_rounds", 0)
        and summary.get("oracle_rounds", 0)
        and summary.get("random_rounds", 0)
        and summary["oracle_accepted"] >= summary["live_accepted"]
        and summary["oracle_accepted"] > summary["random_accepted"]
    ):
        lines.append("Repairs that succeeded line up with the norm shortcut.")
    if summary["later_rounds"] > 0 and summary.get("frozen_rounds", 0) > 0:
        if summary["live_later_accepted"] > summary["frozen_later_accepted"]:
            lines.append("Later rounds favor the updated model over the frozen original editor.")
        else:
            lines.append("Later rounds keep the frozen original editor level with, or ahead of, the updated model.")
    return " ".join(lines)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SELF-EDIT controlled self-repair experiment")
    sub = parser.add_subparsers(dest="cmd", required=True)
    selftest = sub.add_parser("selftest", help="check the harness on a tiny map")
    selftest.add_argument("--out", type=Path, default=None)
    m3 = sub.add_parser("m3", help="plant a scaled column inside a live language model")
    m3.add_argument("--model", default=DEFAULT_MODEL)
    m3.add_argument("--rounds", type=int, default=3)
    m3.add_argument("--seed", type=int, default=0)
    m3.add_argument("--device", default="auto")
    m3.add_argument("--conditions", default=",".join(CONDITIONS))
    m3.add_argument("--out", type=Path, default=None)
    sweep = sub.add_parser("e3e4", help="run the E3 staircase and the E4 information ladder")
    sweep.add_argument("--model", default=DEFAULT_MODEL)
    sweep.add_argument("--device", default="auto")
    sweep.add_argument("--no-live", action="store_true")
    sweep.add_argument("--out", type=Path, default=None)
    programme = sub.add_parser("programme", help="run E5 through E37")
    programme.add_argument("--model", default=DEFAULT_MODEL)
    programme.add_argument("--device", default="auto")
    programme.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.cmd == "selftest":
        run_selftest(args.out)
        return
    if args.cmd == "programme":
        from selfedit.programme import run_programme

        run_programme(model_id=args.model, device=args.device, out_dir=args.out)
        return
    if args.cmd == "e3e4":
        from selfedit.ladder import run_e3_e4

        run_e3_e4(
            model_id=args.model,
            device=args.device,
            live=not args.no_live,
            out_dir=args.out,
        )
        return
    conditions = tuple(part.strip() for part in args.conditions.split(",") if part.strip())
    try:
        run_m3(
            model_id=args.model,
            rounds=args.rounds,
            seed=args.seed,
            conditions=conditions,
            device=args.device,
            out_dir=args.out,
        )
    except KeyboardInterrupt:
        sys.exit(130)
