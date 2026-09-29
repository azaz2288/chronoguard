"""Paired, circular moving-block bootstrap for held-out loss differences."""

from __future__ import annotations

import math
import random
from typing import Any

from .common import InputError


def loss_comparison(predictions: list[dict[str, Any]], repetitions: int = 1000,
                    block_size: int = 1, seed: int = 0) -> dict[str, Any]:
    if type(repetitions) is not int or not 100 <= repetitions <= 10000:
        raise InputError("bootstrap-reps must be an integer between 100 and 10000")
    if type(block_size) is not int or block_size < 1:
        raise InputError("block-size must be a positive integer")
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise InputError("seed must be an integer from 0 to 4294967295")
    n = len(predictions)
    if n == 0:
        raise InputError("Cannot compare losses without held-out predictions")
    if block_size > n:
        raise InputError("block-size cannot exceed the number of held-out predictions")
    deltas = [(row["prediction"] - row["target"]) ** 2 -
              (row["train_mean_baseline"] - row["target"]) ** 2 for row in predictions]
    if not all(math.isfinite(value) for value in deltas):
        raise InputError("Non-finite paired loss differences")
    try:
        average = math.fsum(deltas) / n
    except OverflowError as exc:
        raise InputError("Paired loss differences overflow") from exc
    if not math.isfinite(average):
        raise InputError("Non-finite mean paired loss difference")
    result = {"method": "circular-moving-block-bootstrap", "loss": "squared-error",
              "delta": "model-minus-train-mean-baseline", "count": n,
              "mean_loss_delta": average, "repetitions": repetitions,
              "block_size": block_size, "seed": seed, "interval_95": None}
    if n < 10:
        result["interval_unavailable_reason"] = "fewer_than_10_held_out_predictions"
        return result
    rng = random.Random(seed)
    estimates = []
    blocks = (n + block_size - 1) // block_size
    for _ in range(repetitions):
        indices = []
        for _ in range(blocks):
            start = rng.randrange(n)
            indices.extend((start + offset) % n for offset in range(block_size))
        try:
            estimate = math.fsum(deltas[index] for index in indices[:n]) / n
        except OverflowError as exc:
            raise InputError("Bootstrap loss estimate overflows") from exc
        if not math.isfinite(estimate):
            raise InputError("Non-finite bootstrap loss estimate")
        estimates.append(estimate)
    estimates.sort()
    lower = estimates[math.floor(0.025 * (repetitions - 1))]
    upper = estimates[math.ceil(0.975 * (repetitions - 1))]
    result["interval_95"] = [lower, upper]
    return result
