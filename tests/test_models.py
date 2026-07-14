import numpy as np
import pandas as pd
import pytest

from src.models.models import (
    build_model,
    MODELS,
    BaseModel,
    TreeModel,
    LinearModel,
    XGBoostModel,
    LogRegModel,
)


def test_registry_has_all_models():
    assert set(MODELS) == {"xgboost", "catboost", "lightgbm", "log_reg"}


def test_build_model_returns_correct_subclass():
    assert isinstance(build_model("xgboost", {}), XGBoostModel)
    assert isinstance(build_model("log_reg", {}), LogRegModel)
    assert isinstance(build_model("xgboost", {}), BaseModel)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        build_model("not_a_model", {})


def test_name_attribute():
    assert build_model("xgboost", {}).name == "xgboost"
    assert build_model("log_reg", {}).name == "log_reg"


def test_model_hierarchy():
    # tree models share the TreeExplainer path, linear models the LinearExplainer.
    assert issubclass(XGBoostModel, TreeModel)
    assert issubclass(LogRegModel, LinearModel)
    assert issubclass(TreeModel, BaseModel)
    assert issubclass(LinearModel, BaseModel)


def test_fit_predict_predict_proba():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 4)), columns=[f"f{i}" for i in range(4)])
    y = (rng.random(60) < 0.4).astype(int)

    model = build_model("log_reg", {"max_iter": 1000})
    assert model.fit(X, y) is model  # fit returns self

    y_pred = model.predict(X)
    assert set(np.unique(y_pred)) <= {0, 1}

    y_score = model.predict_proba(X)
    assert y_score.shape == (60,)  # positive-class probability, 1-D
    assert y_score.min() >= 0 and y_score.max() <= 1
