import pytest
from sklearn.metrics import average_precision_score

from src.services.metrics import evaluate_model


def test_keys_and_basic_values():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 0]  # tp=1, fp=1, fn=1, tn=1
    y_score = [0.1, 0.4, 0.9, 0.3]
    m = evaluate_model(y_true, y_pred, y_score)
    assert set(m) == {"precision", "recall", "f1_score", "pr_auc"}
    assert m["precision"] == pytest.approx(0.5)
    assert m["recall"] == pytest.approx(0.5)
    assert m["f1_score"] == pytest.approx(0.5)


def test_pr_auc_uses_scores_not_labels():
    # y_score ranks the positives perfectly; y_pred does not. pr_auc must follow
    # the scores (the bug we fixed was passing y_pred into average_precision).
    y_true = [0, 0, 1, 1]
    y_pred = [1, 0, 0, 1]
    y_score = [0.1, 0.2, 0.8, 0.9]
    m = evaluate_model(y_true, y_pred, y_score)

    ap_scores = average_precision_score(y_true, y_score)
    ap_labels = average_precision_score(y_true, y_pred)
    assert m["pr_auc"] == pytest.approx(ap_scores)
    assert ap_scores != pytest.approx(ap_labels)  # the distinction actually matters


def test_zero_division_returns_zero_not_crash():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 0, 0]  # no positive predictions
    y_score = [0.1, 0.2, 0.3, 0.4]
    m = evaluate_model(y_true, y_pred, y_score)
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["f1_score"] == 0.0
