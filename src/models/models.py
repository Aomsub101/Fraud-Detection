from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod
import logging
import shap

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    name = "base"

    def __init__(self, params=None):
        self.params = params or {}
        self.model = self._build_estimator(self.params)

    @abstractmethod
    def _build_estimator(self, params):
        """Return the underlying estimator"""

    @abstractmethod
    def fit(self, X, y):
        logger.info("Training %s on %d samples", self.name, len(X))
        self.model.fit(X, y)
        logger.info("Finished training %s", self.name)
        return self

    @abstractmethod
    def predict(self, X):
        logger.info("Predicting labels with %s on %d samples", self.name, len(X))
        return self.model.predict(X)

    @abstractmethod
    def predict_proba(self, X):
        logger.info("Predicting probabilities with %s on %d samples", self.name, len(X))
        return self.model.predict_proba(X)[:, 1]

    @abstractmethod
    def shap(self, X):
        ...


class XGBoostModel(BaseModel):
    name = "xgboost"

    def fit(self, X, y):
        logger.info("Training %s on %d samples", self.name, len(X))
        self.model.fit(X, y)
        logger.info("Finished training %s", self.name)
        return self

    def predict(self, X):
        logger.info("Predicting labels with %s on %d samples", self.name, len(X))
        return self.model.predict(X)

    def predict_proba(self, X):
        logger.info("Predicting probabilities with %s on %d samples", self.name, len(X))
        return self.model.predict_proba(X)[:, 1]

    def _build_estimator(self, params):
        return XGBClassifier(**params)


class CatBoostModel(BaseModel):
    name = "catboost"

    def fit(self, X, y):
        logger.info("Training %s on %d samples", self.name, len(X))
        self.model.fit(X, y)
        logger.info("Finished training %s", self.name)
        return self

    def predict(self, X):
        logger.info("Predicting labels with %s on %d samples", self.name, len(X))
        return self.model.predict(X)

    def predict_proba(self, X):
        logger.info("Predicting probabilities with %s on %d samples", self.name, len(X))
        return self.model.predict_proba(X)[:, 1]

    def _build_estimator(self, params):
        return CatBoostClassifier(**params)


class LightGBMModel(BaseModel):
    name = "lightgbm"

    def fit(self, X, y):
        logger.info("Training %s on %d samples", self.name, len(X))
        self.model.fit(X, y)
        logger.info("Finished training %s", self.name)
        return self

    def predict(self, X):
        logger.info("Predicting labels with %s on %d samples", self.name, len(X))
        return self.model.predict(X)

    def predict_proba(self, X):
        logger.info("Predicting probabilities with %s on %d samples", self.name, len(X))
        return self.model.predict_proba(X)[:, 1]

    def _build_estimator(self, params):
        return LGBMClassifier(**params)


class LogRegModel(BaseModel):
    name = "log_reg"

    def fit(self, X, y):
        logger.info("Training %s on %d samples", self.name, len(X))
        self.model.fit(X, y)
        logger.info("Finished training %s", self.name)
        return self

    def predict(self, X):
        logger.info("Predicting labels with %s on %d samples", self.name, len(X))
        return self.model.predict(X)

    def predict_proba(self, X):
        logger.info("Predicting probabilities with %s on %d samples", self.name, len(X))
        return self.model.predict_proba(X)[:, 1]

    def _build_estimator(self, params):
        return LogisticRegression(**params)


MODELS = {cls.name: cls for cls in (XGBoostModel, CatBoostModel, LightGBMModel, LogRegModel)}


def build_model(name, params=None):
    logger.info("Building model '%s' with params %s", name, params)
    return MODELS[name](params)
