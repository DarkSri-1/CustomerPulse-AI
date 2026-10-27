"""
Stage 8: Model Evaluation & Threshold Optimization Module for CustomerPulse AI.
Computes Precision, Recall, F1, PR-AUC, ROC-AUC, Confusion Matrices, optimal
probability thresholds, and saves diagnostic evaluation charts.
"""
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

from src.config import (
    FIGURES_DIR,
    REPORTS_DIR,
    MODEL_REPORT_JSON_PATH,
    MODEL_REPORT_MD_PATH,
)

logger = logging.getLogger(__name__)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Calculate comprehensive classification metrics at a given probability threshold.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "decision_threshold": round(float(threshold), 4),
        "precision_score": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall_score": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_prob)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_prob)), 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    oof_probs: np.ndarray,
    min_recall: float = 0.65,
) -> Tuple[float, Dict[str, float]]:
    """
    Search probability thresholds in [0.20, 0.80] on Out-of-Fold (OOF) validation
    probabilities to maximize F1-score while maintaining recall >= min_recall.
    """
    thresholds = np.linspace(0.20, 0.80, 121)
    best_thresh = 0.50
    best_f1 = -1.0
    best_stats = {}

    for thresh in thresholds:
        y_pred = (oof_probs >= thresh).astype(int)
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))

        if rec >= min_recall and f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thresh)
            best_stats = {"threshold": round(best_thresh, 4), "f1": round(f1, 4), "recall": round(rec, 4), "precision": round(prec, 4)}

    # Fallback to unconstrained max F1 if min_recall was not met
    if best_f1 < 0:
        for thresh in thresholds:
            y_pred = (oof_probs >= thresh).astype(int)
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(thresh)
                best_stats = {"threshold": round(best_thresh, 4), "f1": round(f1, 4)}

    return round(best_thresh, 4), best_stats


def plot_model_comparisons(
    y_test: np.ndarray,
    test_probs_by_model: Dict[str, np.ndarray],
    best_model_name: str,
    best_threshold: float,
) -> None:
    """
    Generate and save ROC + Precision-Recall comparison curves and threshold diagnostics.
    """
    # Chart 6: ROC Curve & Precision-Recall Curve side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    palette = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728"]

    for idx, (model_name, probs) in enumerate(test_probs_by_model.items()):
        color = palette[idx % len(palette)]
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = roc_auc_score(y_test, probs)
        axes[0].plot(fpr, tpr, linewidth=2.2, label=f"{model_name} (ROC-AUC = {roc_auc:.3f})", color=color)

        prec, rec, _ = precision_recall_curve(y_test, probs)
        pr_auc = average_precision_score(y_test, probs)
        axes[1].plot(rec, prec, linewidth=2.2, label=f"{model_name} (PR-AUC = {pr_auc:.3f})", color=color)

    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", alpha=0.5)
    axes[0].set_title("ROC Curve Comparison (Holdout Test Set)", fontweight="bold")
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(loc="lower right")

    baseline_pos_rate = float(np.mean(y_test))
    axes[1].axhline(baseline_pos_rate, linestyle="--", color="black", alpha=0.5, label=f"Baseline Churn Rate ({baseline_pos_rate:.2f})")
    axes[1].set_title("Precision-Recall Curve Comparison (Holdout Test Set)", fontweight="bold")
    axes[1].set_xlabel("Recall (Sensitivity)")
    axes[1].set_ylabel("Precision (Positive Predictive Value)")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "06_model_comparison_roc_pr_curves.png")
    plt.close(fig)

    # Chart 7: Confusion Matrix of Best Model & Threshold Trade-off Curve
    best_probs = test_probs_by_model[best_model_name]
    y_pred_best = (best_probs >= best_threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred_best, labels=[0, 1])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Retained (0)", "Predicted Churned (1)"],
        yticklabels=["Actual Retained (0)", "Actual Churned (1)"],
        ax=axes[0],
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[0].set_title(f"{best_model_name} Confusion Matrix (Threshold = {best_threshold:.2f})", fontweight="bold")

    thresholds = np.linspace(0.15, 0.85, 71)
    prec_list, rec_list, f1_list = [], [], []
    for t in thresholds:
        p_t = (best_probs >= t).astype(int)
        prec_list.append(precision_score(y_test, p_t, zero_division=0))
        rec_list.append(recall_score(y_test, p_t, zero_division=0))
        f1_list.append(f1_score(y_test, p_t, zero_division=0))

    axes[1].plot(thresholds, prec_list, label="Precision", linewidth=2, color="#1f77b4")
    axes[1].plot(thresholds, rec_list, label="Recall", linewidth=2, color="#2ca02c")
    axes[1].plot(thresholds, f1_list, label="F1-Score", linewidth=2.5, color="#d62728")
    axes[1].axvline(best_threshold, linestyle="--", color="black", label=f"Selected Threshold ({best_threshold:.2f})")
    axes[1].set_title(f"{best_model_name}: Precision, Recall & F1 vs. Decision Threshold", fontweight="bold")
    axes[1].set_xlabel("Probability Decision Threshold")
    axes[1].set_ylabel("Metric Score")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "07_confusion_matrix_and_threshold_curve.png")
    plt.close(fig)
    logger.info(f"Saved Stage 8 evaluation figures to {FIGURES_DIR}")


def save_evaluation_reports(report_payload: Dict[str, Any]) -> None:
    """Save complete Stage 8 evaluation results to JSON and Markdown."""
    with open(MODEL_REPORT_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4)

    best_model = report_payload["best_model_name"]
    best_thresh = report_payload["best_decision_threshold"]
    test_rows: List[Dict[str, Any]] = report_payload["holdout_test_metrics"]
    cv_rows: List[Dict[str, Any]] = report_payload["cv_oof_metrics"]

    md = f"""# Stage 8: Machine Learning Model Evaluation Summary — CustomerPulse AI

## 1. Selected Production Model
- **Winning Model:** `{best_model}` (Version `{report_payload['model_version']}`)
- **Selection Criterion:** Highest Holdout Test **PR-AUC** & **F1-Score**
- **Optimized Decision Threshold (from 5-Fold OOF Validation):** `{best_thresh:.4f}`
- **Train / Holdout Split:** `{report_payload['train_size']:,}` Train Customers (80%) | `{report_payload['test_size']:,}` Holdout Test Customers (20%)

## 2. Holdout Test Set Performance Comparison (Unseen Customers)
| Model | Threshold | PR-AUC | ROC-AUC | F1-Score | Recall | Precision | TP | FP | TN | FN |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for r in test_rows:
        star = " **" if r["model_name"] == best_model else ""
        md += (
            f"| {star}{r['model_name']}{star} | {r['decision_threshold']:.2f} | "
            f"{r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['f1_score']:.4f} | "
            f"{r['recall_score']:.4f} | {r['precision_score']:.4f} | "
            f"{r['true_positives']} | {r['false_positives']} | {r['true_negatives']} | {r['false_negatives']} |\n"
        )

    md += """
## 3. 5-Fold Stratified Cross-Validation (Out-of-Fold Training Set)
| Model | Threshold | OOF PR-AUC | OOF ROC-AUC | OOF F1-Score | OOF Recall | OOF Precision |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
"""
    for r in cv_rows:
        md += (
            f"| {r['model_name']} | {r['decision_threshold']:.2f} | "
            f"{r['pr_auc']:.4f} | {r['roc_auc']:.4f} | {r['f1_score']:.4f} | "
            f"{r['recall_score']:.4f} | {r['precision_score']:.4f} |\n"
        )

    with open(MODEL_REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)

    logger.info(f"Saved evaluation JSON report to     : {MODEL_REPORT_JSON_PATH}")
    logger.info(f"Saved evaluation Markdown report to : {MODEL_REPORT_MD_PATH}")
