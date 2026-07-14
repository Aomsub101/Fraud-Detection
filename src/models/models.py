from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod
import shap


class BaseModel(ABC):
    name = "base"

    def __init__(self, params=None):
        self.params = params or {}
        self.model = self._build_estimator(self.params)

    @abstractmethod
    def _build_estimator(self, params):
        """Return the underlying estimator"""

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def shap(self, X):
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)

    @property
    def __name__(self):
        return self.name


class XGBoostModel(BaseModel):
    name = "xgboost"

    def _build_estimator(self, params):
        return XGBClassifier(**params)


class CatBoostModel(BaseModel):
    name = "catboost"

    def _build_estimator(self, params):
        return CatBoostClassifier(**params)


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def _build_estimator(self, params):
        return LGBMClassifier(**params)


class LogRegModel(BaseModel):
    name = "log_reg"

    def _build_estimator(self, params):
        return LogisticRegression(**params)

    def shap(self, X):
        return shap.LinearExplainer(self.model, X).shap_values(X)


MODELS = {cls.name: cls for cls in (XGBoostModel, CatBoostModel, LightGBMModel, LogRegModel)}


def build_model(name, params=None):
    return MODELS[name](params)
