"""Run the fully audited synthetic demonstration without leaving artifacts."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path


SOURCE = Path(__file__).resolve().parent / "pipeline"


def run(*arguments: object) -> None:
    command = [sys.executable, "-m", "chronoguard", *(str(value) for value in arguments)]
    subprocess.run(command, check=True)


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        joined = root / "joined.csv"
        observations = root / "observations.csv"
        plan = root / "plan.json"
        report = root / "report.json"
        samples, facts, targets, events = (SOURCE / name for name in
                                           ("samples.csv", "facts.csv", "targets.csv", "events.csv"))
        run("join", samples, facts, joined, "--max-age-hours", 24)
        run("audit-join", samples, facts, joined, "--max-age-hours", 24)
        run("assemble", samples, facts, joined, targets, observations, "--max-age-hours", 24)
        run("split", events, plan, "--folds", 2, "--min-train-times", 10, "--test-times", 5)
        run("audit", events, plan)
        run("evaluate", events, plan, observations, report, "--feature", "signal",
            "--bootstrap-reps", 500, "--block-size", 3, "--seed", 23)
        run("audit-evaluation", events, plan, observations, report)
        result = json.loads(report.read_text(encoding="utf-8"))
        print("Held-out RMSE:", result["overall"]["rmse"])
        print("Train-mean baseline RMSE:", result["overall_baseline"]["rmse"])
        print("Paired loss-delta interval:", result["loss_comparison"]["interval_95"])


if __name__ == "__main__":
    main()
