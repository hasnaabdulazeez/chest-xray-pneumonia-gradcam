"""Clinical-style metrics and operating-point selection (numpy only).

For a screening tool, missing a pneumonia case (false negative) is worse than a
false alarm, so instead of the default 0.5 cut-off the decision threshold is
chosen on the validation set to reach a target sensitivity, and specificity is
reported at that operating point.
"""

from __future__ import annotations

import numpy as np


def confusion(y_true, y_prob, threshold: float) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    return {
        "tp": int(((y_pred == 1) & (y_true == 1)).sum()),
        "tn": int(((y_pred == 0) & (y_true == 0)).sum()),
        "fp": int(((y_pred == 1) & (y_true == 0)).sum()),
        "fn": int(((y_pred == 0) & (y_true == 1)).sum()),
    }


def _safe(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def clinical_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, float]:
    c = confusion(y_true, y_prob, threshold)
    sens = _safe(c["tp"], c["tp"] + c["fn"])
    spec = _safe(c["tn"], c["tn"] + c["fp"])
    ppv = _safe(c["tp"], c["tp"] + c["fp"])
    return {
        "threshold": threshold,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": _safe(c["tn"], c["tn"] + c["fn"]),
        "f1": _safe(2 * ppv * sens, ppv + sens),
        "accuracy": _safe(c["tp"] + c["tn"], sum(c.values())),
        **{k: float(v) for k, v in c.items()},
    }


def threshold_for_sensitivity(y_true, y_prob, target: float = 0.95) -> float:
    """Highest threshold whose sensitivity is at least ``target`` (maximises specificity)."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = np.unique(y_prob)[::-1]  # high -> low
    for t in candidates:
        if clinical_metrics(y_true, y_prob, float(t))["sensitivity"] >= target:
            return float(t)
    return float(candidates[-1]) if len(candidates) else 0.5
