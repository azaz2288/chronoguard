"""Leakage-resistant, reproducible walk-forward ridge baseline."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import InputError, InputSnapshot, read_csv
from .split import Event, audit_plan
from .statistics import loss_comparison


@dataclass(frozen=True)
class Observation:
    sample_id: str
    target: float
    features: tuple[float | None, ...]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise InputError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def load_observations(path: Path | InputSnapshot, features: list[str]) -> dict[str, Observation]:
    if not features or len(set(features)) != len(features) or any(name in ("sample_id", "target") for name in features):
        raise InputError("Specify distinct numeric --feature columns other than sample_id and target")
    observations = {}
    for number, row in enumerate(read_csv(path, ("sample_id", "target", *features)), start=2):
        sample_id = row["sample_id"].strip()
        if not sample_id or sample_id in observations:
            raise InputError(f"{path}:{number}: empty or duplicate sample_id")
        try:
            target = float(row["target"])
            values = tuple(float(row[name]) if row[name].strip() else None for name in features)
        except ValueError as exc:
            raise InputError(f"{path}:{number}: target and features must be numeric") from exc
        if not math.isfinite(target) or any(value is not None and not math.isfinite(value) for value in values):
            raise InputError(f"{path}:{number}: non-finite target or feature")
        observations[sample_id] = Observation(sample_id, target, values)
    if not observations:
        raise InputError(f"{path}: no observations")
    return observations


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    """Partial-pivot Gaussian elimination on a regularized normal system."""
    size = len(vector)
    augmented = [row[:] + [value] for row, value in zip(matrix, vector)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda index: abs(augmented[index][column]))
        if augmented[pivot][column] == 0:
            raise InputError("Ridge system is singular; increase alpha")
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        for index in range(column + 1, size):
            factor = augmented[index][column] / divisor
            for offset in range(column, size + 1):
                augmented[index][offset] -= factor * augmented[column][offset]
    solution = [0.0] * size
    for column in range(size - 1, -1, -1):
        solution[column] = (augmented[column][size] - sum(augmented[column][j] * solution[j] for j in range(column + 1, size))) / augmented[column][column]
    if not all(math.isfinite(value) for value in solution):
        raise InputError("Non-finite ridge coefficients; inspect feature scale")
    return solution


def _fit(train: list[Observation], feature_names: list[str], alpha: float) -> dict[str, Any]:
    dimension = len(feature_names)
    means = []
    scales = []
    for index in range(dimension):
        available = [row.features[index] for row in train if row.features[index] is not None]
        if not available:
            raise InputError(f"Training fold has no observed values for feature {feature_names[index]!r}")
        mean = math.fsum(available) / len(available)
        variance = math.fsum((value - mean) ** 2 for value in available) / len(available)
        scale = math.sqrt(variance)
        if not math.isfinite(mean) or not math.isfinite(scale):
            raise InputError(f"Non-finite training statistics for {feature_names[index]!r}")
        means.append(mean)
        scales.append(scale if scale > 0 else 1.0)
    centered_target = math.fsum(row.target for row in train) / len(train)
    transformed = [_transform(row, means, scales) for row in train]
    gram = [[math.fsum(vector[i] * vector[j] for vector in transformed) + (alpha if i == j else 0.0)
             for j in range(dimension)] for i in range(dimension)]
    right = [math.fsum(vector[i] * (row.target - centered_target) for vector, row in zip(transformed, train)) for i in range(dimension)]
    coefficients = _solve(gram, right)
    return {"impute_means": means, "scales": scales, "intercept": centered_target, "coefficients": coefficients, "alpha": alpha}


def _transform(row: Observation, means: list[float], scales: list[float]) -> list[float]:
    values = [0.0 if value is None else (value - means[index]) / scales[index]
              for index, value in enumerate(row.features)]
    if not all(math.isfinite(value) for value in values):
        raise InputError(f"Non-finite transformed features for {row.sample_id!r}")
    return values


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    count = len(rows)
    errors = [row["prediction"] - row["target"] for row in rows]
    squared = [error * error for error in errors]
    truth_mean = math.fsum(row["target"] for row in rows) / count
    denominator = math.fsum((row["target"] - truth_mean) ** 2 for row in rows)
    result = {
        "count": count,
        "mae": math.fsum(abs(error) for error in errors) / count,
        "rmse": math.sqrt(math.fsum(squared) / count),
        "r2": 1 - math.fsum(squared) / denominator if denominator else None,
    }
    if any(isinstance(value, float) and not math.isfinite(value) for value in result.values()):
        raise InputError("Non-finite evaluation metrics; inspect target scale")
    return result


def evaluate(events: list[Event], plan: dict[str, Any], observations: dict[str, Observation], features: list[str], alpha: float,
             source_hashes: dict[str, str], bootstrap_reps: int = 1000, block_size: int = 1, seed: int = 0) -> dict[str, Any]:
    if not math.isfinite(alpha) or alpha <= 0:
        raise InputError("alpha must be positive and finite")
    issues = audit_plan(events, plan)
    if issues:
        raise InputError("Split plan failed audit: " + "; ".join(issues[:5]))
    event_ids = {event.sample_id for event in events}
    if set(observations) != event_ids:
        missing = sorted(event_ids - set(observations))
        extra = sorted(set(observations) - event_ids)
        raise InputError(f"Observation IDs do not match events; missing={missing[:5]}, extra={extra[:5]}")
    held_out_count = sum(len(fold["test_ids"]) for fold in plan["folds"])
    if block_size > held_out_count:
        raise InputError("block-size cannot exceed the number of held-out predictions")
    folds = []
    all_predictions = []
    for fold in plan["folds"]:
        train = [observations[sample_id] for sample_id in fold["train_ids"]]
        model = _fit(train, features, alpha)
        predictions = []
        for sample_id in fold["test_ids"]:
            row = observations[sample_id]
            vector = _transform(row, model["impute_means"], model["scales"])
            prediction = model["intercept"] + math.fsum(coefficient * value for coefficient, value in zip(model["coefficients"], vector))
            if not math.isfinite(prediction):
                raise InputError(f"Non-finite prediction for {sample_id!r}")
            predictions.append({"sample_id": sample_id, "target": row.target, "prediction": prediction,
                                "train_mean_baseline": model["intercept"]})
        metrics = _metrics(predictions)
        baseline_metrics = _metrics([{"target": row["target"], "prediction": row["train_mean_baseline"]} for row in predictions])
        all_predictions.extend(predictions)
        folds.append({"fold": fold["fold"], "train_count": len(train), "purged_count": len(fold["purged_ids"]),
                      "test_start": fold["test_start"], "model": model, "metrics": metrics,
                      "baseline_metrics": baseline_metrics, "predictions": predictions})
    return {"version": 2, "method": "train-only-imputed-standardized-ridge", "features": features,
            "source_sha256": source_hashes, "folds": folds, "overall": _metrics(all_predictions),
            "overall_baseline": _metrics([{"target": row["target"], "prediction": row["train_mean_baseline"]}
                                          for row in all_predictions]),
            "loss_comparison": loss_comparison(all_predictions, bootstrap_reps, block_size, seed)}
