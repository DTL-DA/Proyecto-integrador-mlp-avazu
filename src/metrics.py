"""Metricas de evaluacion compartidas por scikit-learn y PySpark (sobre arrays numpy)."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
    accuracy_score,
)


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def calibration_curve_data(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, bins[1:-1])
    mean_pred, frac_pos = [], []
    for b in range(n_bins):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        mean_pred.append(y_prob[mask].mean())
        frac_pos.append(y_true[mask].mean())
    return np.array(mean_pred), np.array(frac_pos)


def choose_threshold_on_validation(y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Umbral que maximiza F1 en validacion; se congela y se usa una sola vez en prueba.

    Vectorizado sobre precision_recall_curve (que ya ordena por umbral internamente)
    en vez de recorrer cada umbral unico en Python y recalcular f1_score sobre todo
    el set de validacion en cada paso -- con ~2M filas de validacion ese bucle era
    O(n_umbrales x n), el cuello de botella real de cada corrida.
    """
    precision, recall, thresholds = precision_recall_curve(y_val, p_val)
    precision, recall = precision[:-1], recall[:-1]  # el ultimo punto no tiene umbral
    denom = precision + recall
    f1 = np.where(denom > 0, 2 * precision * recall / np.where(denom == 0, 1.0, denom), 0.0)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def lift_and_gain(y_true: np.ndarray, y_prob: np.ndarray, n_deciles: int = 10):
    order = np.argsort(-y_prob)
    y_sorted = y_true[order]
    n = len(y_true)
    base_rate = y_true.mean()
    rows = []
    cum_positives = 0
    for d in range(1, n_deciles + 1):
        end = int(n * d / n_deciles)
        cum_positives = y_sorted[:end].sum()
        decile_rate = cum_positives / end
        rows.append(
            {
                "decil": d,
                "cobertura": end / n,
                "ganancia_acumulada": cum_positives / y_true.sum(),
                "lift": decile_rate / base_rate,
            }
        )
    return rows


def full_metric_report(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_ci(values: np.ndarray, n_resamples: int = 1000, alpha: float = 0.05, seed: int = 13) -> dict:
    rng = np.random.default_rng(seed)
    n = len(values)
    boot_means = np.array([
        rng.choice(values, size=n, replace=True).mean() for _ in range(n_resamples)
    ])
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)) if n > 1 else 0.0,
            "ci_low": float(lo), "ci_high": float(hi), "n": n}
