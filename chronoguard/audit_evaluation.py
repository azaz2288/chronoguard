"""Independent checks for a saved walk-forward evaluation report.

This module does not call the experiment runner's fit, transform or metric
functions. It checks the saved model against training data and ridge normal
equations, then recomputes held-out predictions and metrics.
"""

from __future__ import annotations

import math
from typing import Any

from .common import InputError
from .evaluate import Observation
from .split import Event, audit_plan
from .statistics import loss_comparison


def _number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise InputError(f"{label} must be a finite number")
    return float(value)


def _vector(value: Any, size: int, label: str) -> list[float]:
    if not isinstance(value, list) or len(value) != size:
        raise InputError(f"{label} must have {size} numbers")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _close(actual: float, expected: float) -> bool:
    return math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-10)


def _metric_values(rows: list[tuple[float, float]]) -> dict[str, float | int | None]:
    count = len(rows)
    errors = [prediction - target for target, prediction in rows]
    mean = math.fsum(target for target, _ in rows) / count
    total = math.fsum((target - mean) ** 2 for target, _ in rows)
    squared = math.fsum(error * error for error in errors)
    return {"count": count, "mae": math.fsum(abs(error) for error in errors) / count,
            "rmse": math.sqrt(squared / count), "r2": 1 - squared / total if total else None}


def _check_metrics(saved: Any, rows: list[tuple[float, float]], label: str, issues: list[str]) -> None:
    if not isinstance(saved, dict):
        raise InputError(f"{label} must be an object")
    expected = _metric_values(rows)
    for name, value in expected.items():
        present = saved.get(name)
        if value is None:
            if present is not None:
                issues.append(f"{label}.{name} should be null")
        elif name == "count":
            if type(present) is not int or present != value:
                issues.append(f"{label}.{name} mismatch")
        elif not _close(_number(present, f"{label}.{name}"), value):
            issues.append(f"{label}.{name} mismatch")


def audit_evaluation(events: list[Event], plan: Any, observations: dict[str, Observation],
                     report: Any, source_hashes: dict[str, str]) -> list[str]:
    """Return content violations; malformed inputs raise InputError."""
    plan_issues = audit_plan(events, plan)
    if plan_issues:
        raise InputError("Split plan failed audit: " + "; ".join(plan_issues[:5]))
    if not isinstance(report, dict) or report.get("version") != 2 or report.get("method") != "train-only-imputed-standardized-ridge":
        raise InputError("Unsupported evaluation report format")
    features = report.get("features")
    if not isinstance(features, list) or not features or any(not isinstance(name, str) or not name for name in features) or len(set(features)) != len(features):
        raise InputError("Report features must be a nonempty list of distinct names")
    if not isinstance(report.get("folds"), list) or len(report["folds"]) != len(plan["folds"]):
        raise InputError("Report fold count does not match the plan")
    if set(observations) != {event.sample_id for event in events}:
        raise InputError("Observation IDs do not match events")
    if any(len(row.features) != len(features) for row in observations.values()):
        raise InputError("Observation feature width does not match report")
    issues = []
    if report.get("source_sha256") != source_hashes:
        issues.append("Source SHA-256 fingerprints do not match current inputs")
    all_model_rows: list[tuple[float, float]] = []
    all_baseline_rows: list[tuple[float, float]] = []
    first_alpha: float | None = None
    for index, (planned, saved) in enumerate(zip(plan["folds"], report["folds"]), start=1):
        label = f"Fold {index}"
        if not isinstance(saved, dict):
            raise InputError(f"{label} report entry must be an object")
        for field, expected in (("fold", planned["fold"]), ("train_count", len(planned["train_ids"])),
                                ("purged_count", len(planned["purged_ids"])), ("test_start", planned["test_start"])):
            if saved.get(field) != expected:
                issues.append(f"{label}: {field} mismatch")
        model = saved.get("model")
        if not isinstance(model, dict):
            raise InputError(f"{label} model must be an object")
        dimension = len(features)
        means = _vector(model.get("impute_means"), dimension, f"{label} impute_means")
        scales = _vector(model.get("scales"), dimension, f"{label} scales")
        coefficients = _vector(model.get("coefficients"), dimension, f"{label} coefficients")
        intercept = _number(model.get("intercept"), f"{label} intercept")
        alpha = _number(model.get("alpha"), f"{label} alpha")
        if alpha <= 0 or any(scale <= 0 for scale in scales):
            raise InputError(f"{label}: alpha and scales must be positive")
        if first_alpha is None:
            first_alpha = alpha
        elif alpha != first_alpha:
            issues.append(f"{label}: alpha differs from earlier folds")
        train = [observations[sample_id] for sample_id in planned["train_ids"]]
        train_mean = math.fsum(row.target for row in train) / len(train)
        if not _close(intercept, train_mean):
            issues.append(f"{label}: intercept differs from training target mean")
        for feature_index, name in enumerate(features):
            values = [row.features[feature_index] for row in train if row.features[feature_index] is not None]
            if not values:
                raise InputError(f"{label}: no observed training values for {name!r}")
            mean = math.fsum(values) / len(values)
            standard = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
            scale = standard if standard > 0 else 1.0
            if not _close(means[feature_index], mean):
                issues.append(f"{label}: training imputation mean mismatch for {name}")
            if not _close(scales[feature_index], scale):
                issues.append(f"{label}: training scale mismatch for {name}")
        # Verify the saved coefficients satisfy the ridge normal equations,
        # using independently computed training statistics.
        vectors = [[0.0 if row.features[j] is None else (row.features[j] - means[j]) / scales[j]
                    for j in range(dimension)] for row in train]
        for feature_index, name in enumerate(features):
            left = math.fsum(vector[feature_index] * math.fsum(coefficient * value for coefficient, value in zip(coefficients, vector))
                             for vector in vectors) + alpha * coefficients[feature_index]
            right = math.fsum(vector[feature_index] * (row.target - intercept) for vector, row in zip(vectors, train))
            if not _close(left, right):
                issues.append(f"{label}: ridge normal equation failed for {name}")
        predictions = saved.get("predictions")
        if not isinstance(predictions, list) or len(predictions) != len(planned["test_ids"]):
            raise InputError(f"{label} prediction count does not match plan")
        fold_model_rows: list[tuple[float, float]] = []
        fold_baseline_rows: list[tuple[float, float]] = []
        for sample_id, prediction in zip(planned["test_ids"], predictions):
            if not isinstance(prediction, dict):
                raise InputError(f"{label} prediction must be an object")
            if prediction.get("sample_id") != sample_id:
                issues.append(f"{label}: prediction ID/order mismatch for {sample_id}")
            row = observations[sample_id]
            observed_target = _number(prediction.get("target"), f"{label} {sample_id} target")
            observed_prediction = _number(prediction.get("prediction"), f"{label} {sample_id} prediction")
            observed_baseline = _number(prediction.get("train_mean_baseline"), f"{label} {sample_id} baseline")
            if not _close(observed_target, row.target):
                issues.append(f"{label}: target mismatch for {sample_id}")
            vector = [0.0 if value is None else (value - means[j]) / scales[j] for j, value in enumerate(row.features)]
            expected_prediction = intercept + math.fsum(coefficient * value for coefficient, value in zip(coefficients, vector))
            if not _close(observed_prediction, expected_prediction):
                issues.append(f"{label}: prediction mismatch for {sample_id}")
            if not _close(observed_baseline, intercept):
                issues.append(f"{label}: baseline prediction mismatch for {sample_id}")
            fold_model_rows.append((row.target, expected_prediction))
            fold_baseline_rows.append((row.target, train_mean))
        _check_metrics(saved.get("metrics"), fold_model_rows, f"{label} metrics", issues)
        _check_metrics(saved.get("baseline_metrics"), fold_baseline_rows, f"{label} baseline_metrics", issues)
        all_model_rows.extend(fold_model_rows)
        all_baseline_rows.extend(fold_baseline_rows)
    _check_metrics(report.get("overall"), all_model_rows, "Overall metrics", issues)
    _check_metrics(report.get("overall_baseline"), all_baseline_rows, "Overall baseline_metrics", issues)
    comparison = report.get("loss_comparison")
    if not isinstance(comparison, dict):
        raise InputError("Report loss_comparison must be an object")
    repetitions = comparison.get("repetitions")
    block_size = comparison.get("block_size")
    seed = comparison.get("seed")
    combined = [{"target": target, "prediction": model, "train_mean_baseline": baseline}
                for (target, model), (_, baseline) in zip(all_model_rows, all_baseline_rows)]
    expected_comparison = loss_comparison(combined, repetitions, block_size, seed)
    for field, expected in expected_comparison.items():
        actual = comparison.get(field)
        if field == "mean_loss_delta":
            if not _close(_number(actual, f"loss_comparison.{field}"), expected):
                issues.append(f"loss_comparison.{field} mismatch")
        elif field == "interval_95" and expected is not None:
            values = _vector(actual, 2, "loss_comparison.interval_95")
            if any(not _close(item, target) for item, target in zip(values, expected)):
                issues.append("loss_comparison.interval_95 mismatch")
        elif actual != expected:
            issues.append(f"loss_comparison.{field} mismatch")
    return issues
