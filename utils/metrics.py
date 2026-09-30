# ============================================================================
# FILE: utils/metrics.py
# Metric helpers for classification reports, curve data, and formatted summaries.
#
# Purpose:
# - Centralizes metric computation used by the detection pipeline and plotting
#   utilities.
#
# Workflow:
# - Computes confusion matrices and derived statistics.
# - Generates ROC and PR curve data.
# - Builds readable metric tables for terminal output.
#
# Use this file when:
# - You want to verify metric definitions or reporting behavior.
# ============================================================================
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)


def compute_confusion_matrix_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Compute confusion matrix as dictionary
    
    Returns:
        {
            "tn": int,
            "fp": int, 
            "fn": int,
            "tp": int
        }
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle binary classification
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        # Fallback if only one class present
        tn = fp = fn = tp = 0
        if len(np.unique(y_true)) == 1:
            if np.unique(y_true)[0] == 0:
                tn = len(y_true)
            else:
                tp = len(y_true)
    
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp)
    }


def compute_complete_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict:
    """
    Compute all classification metrics
    
    Args:
        y_true: Ground truth labels (0=benign, 1=malicious)
        y_pred: Predicted labels (0=benign, 1=malicious)
        y_prob: Prediction probabilities for class 1 (optional, for AUC)
    
    Returns:
        Complete metrics dictionary
    """
    # Confusion matrix
    cm_dict = compute_confusion_matrix_dict(y_true, y_pred)
    tn, fp, fn, tp = cm_dict["tn"], cm_dict["fp"], cm_dict["fn"], cm_dict["tp"]
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    
    # Error rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # AUC (if probabilities provided)
    auc_roc = None
    if y_prob is not None:
        try:
            auc_roc = roc_auc_score(y_true, y_prob)
        except:
            auc_roc = None
    
    # Average Precision (if probabilities provided)
    avg_precision = None
    if y_prob is not None:
        try:
            avg_precision = average_precision_score(y_true, y_prob)
        except:
            avg_precision = None
    
    return {
        "confusion_matrix": cm_dict,
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "fpr": float(fpr),
        "fnr": float(fnr),
        "auc_roc": float(auc_roc) if auc_roc is not None else None,
        "average_precision": float(avg_precision) if avg_precision is not None else None,
        "support": {
            "benign": int((y_true == 0).sum()),
            "malicious": int((y_true == 1).sum()),
            "total": len(y_true)
        }
    }


def compute_roc_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """
    Compute ROC curve data for visualization
    
    Returns:
        {
            "fpr": List[float],
            "tpr": List[float],
            "thresholds": List[float],
            "auc": float
        }
    """
    try:
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        
        return {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": thresholds.tolist(),
            "auc": float(auc)
        }
    except:
        return None


def compute_precision_recall_curve_data(
    y_true: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """
    Compute Precision-Recall curve data
    
    Returns:
        {
            "precision": List[float],
            "recall": List[float],
            "thresholds": List[float],
            "average_precision": float
        }
    """
    try:
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        avg_prec = average_precision_score(y_true, y_prob)
        
        return {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": thresholds.tolist(),
            "average_precision": float(avg_prec)
        }
    except:
        return None


def format_metrics_table(metrics: Dict) -> str:
    """
    Format metrics as pretty table
    
    Args:
        metrics: Dictionary from compute_complete_metrics()
    
    Returns:
        Formatted string table
    """
    cm = metrics["confusion_matrix"]
    
    table = []
    table.append("=" * 60)
    table.append("CLASSIFICATION METRICS")
    table.append("=" * 60)
    table.append("")
    
    # Confusion Matrix
    table.append("Confusion Matrix:")
    table.append("                 Predicted")
    table.append("              Benign  Malicious")
    table.append(f"   Actual Benign  {cm['tn']:>6}  {cm['fp']:>9}")
    table.append(f"          Malicious  {cm['fn']:>6}  {cm['tp']:>9}")
    table.append("")
    
    # Main Metrics
    table.append("Performance Metrics:")
    table.append(f"   Accuracy:  {metrics['accuracy']:.4f}")
    table.append(f"   Precision: {metrics['precision']:.4f}")
    table.append(f"   Recall:    {metrics['recall']:.4f}")
    table.append(f"   F1 Score:  {metrics['f1_score']:.4f}")
    table.append("")
    
    # Error Rates
    table.append("Error Rates:")
    table.append(f"   FPR (False Positive Rate): {metrics['fpr']:.4f}")
    table.append(f"   FNR (False Negative Rate): {metrics['fnr']:.4f}")
    table.append("")
    
    # AUC if available
    if metrics['auc_roc'] is not None:
        table.append(f"   AUC-ROC: {metrics['auc_roc']:.4f}")
    if metrics['average_precision'] is not None:
        table.append(f"   Avg Precision: {metrics['average_precision']:.4f}")
    
    table.append("=" * 60)
    
    return "\n".join(table)


def compare_metrics(metrics_list: List[Dict], names: List[str]) -> str:
    """
    Compare multiple metrics side by side
    
    Args:
        metrics_list: List of metrics dictionaries
        names: List of names for each metrics set
    
    Returns:
        Formatted comparison table
    """
    table = []
    table.append("=" * 80)
    table.append("METRICS COMPARISON")
    table.append("=" * 80)
    table.append("")
    
    # Header
    header = f"{'Metric':<20}"
    for name in names:
        header += f" {name:>12}"
    table.append(header)
    table.append("-" * 80)
    
    # Metrics rows
    metric_keys = [
        ("Accuracy", "accuracy"),
        ("Precision", "precision"),
        ("Recall", "recall"),
        ("F1 Score", "f1_score"),
        ("FPR", "fpr"),
        ("FNR", "fnr")
    ]
    
    for label, key in metric_keys:
        row = f"{label:<20}"
        for metrics in metrics_list:
            value = metrics.get(key, 0.0)
            row += f" {value:>12.4f}"
        table.append(row)
    
    table.append("=" * 80)
    
    return "\n".join(table)
