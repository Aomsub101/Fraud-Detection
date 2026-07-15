import logging

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def evaluate_model(y_true, y_pred, y_score):
    logger.info("Evaluating model on %d samples", len(y_true))
    fpr, tpr, _ = roc_curve(y_true, y_score)
    metrics = {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "pr_auc": average_precision_score(y_true, y_score),
        "recall@5fpr": tpr[fpr <= 0.05].max(),
    }
    logger.info("Metrics: %s", metrics)
    return metrics
