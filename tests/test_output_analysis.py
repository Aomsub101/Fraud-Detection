import numpy as np
import pytest
from PIL import Image

from src.data.clean_data import clean_data
from src.data.preprocess import preprocess
from src.features.features_engineer import features_engineer
from src.data.data_split import data_split
from src.models.models import build_model
from src.services.metrics import evaluate_model
from src.services.output_analysis import output_analysis


def _nonwhite_pixels(path):
    a = np.asarray(Image.open(path).convert("L"))
    return int((a < 250).sum())


@pytest.mark.slow
def test_output_analysis_writes_nonblank_files(sample_df, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # output/ is written relative to the cwd

    X_train, X_test, y_train, y_test = data_split(
        features_engineer(preprocess(clean_data(sample_df)))
    )
    model = build_model("xgboost", {}).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)
    metrics = evaluate_model(y_test, y_pred, y_score)

    output_analysis(model, metrics, X_test, y_test, y_score)

    out = tmp_path / "output"
    shap_png = out / "shap_plot[xgboost].png"
    pr_png = out / "pr_curve[xgboost].png"
    metrics_json = out / "metrics[xgboost].json"

    assert shap_png.exists()
    assert pr_png.exists()
    assert metrics_json.exists()
    # The SHAP save previously produced a blank image; guard that the plots
    # actually contain rendered content.
    assert _nonwhite_pixels(shap_png) > 500
    assert _nonwhite_pixels(pr_png) > 500
