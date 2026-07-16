from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from abc import ABC, abstractmethod
import logging
import shap
import joblib

logger = logging.getLogger(__name__)
OUTPUT_PATH = "output/"


class BaseModel(ABC):
    name = "base"

    def __init__(self, params=None):
        self.params = params or {}
        self.model = self._build_estimator(self.params)

    @abstractmethod
    def _build_estimator(self, params):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def fit(self, X, y):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def predict(self, X):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def predict_proba(self, X):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def shap(self, X):
        raise NotImplementedError("This method should be overridden by subclasses.")

    @abstractmethod
    def save(self):
        raise NotImplementedError("This method should be overridden by subclasses.")

class XGBoostModel(BaseModel):
    name = "xgboost"

    def _build_estimator(self, params):
        return XGBClassifier(**params)

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

    def shap(self, X):
        logger.info("Computing SHAP values for %s", self.name)
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)
    
    def save(self):
        logger.info("Saving model %s", self.name)
        self.model.save_model(OUTPUT_PATH + "xgboost.json")
        logger.info("Saving done")

class CatBoostModel(BaseModel):
    name = "catboost"

    def _build_estimator(self, params):
        return CatBoostClassifier(**params)

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

    def shap(self, X):
        logger.info("Computing SHAP values for %s", self.name)
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)

    def save(self):
        logger.info("Saving model %s", self.name)
        self.model.save_model(OUTPUT_PATH + "catboost.cbm")
        logger.info("Saving done")

class LightGBMModel(BaseModel):
    name = "lightgbm"

    def _build_estimator(self, params):
        return LGBMClassifier(**params)

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

    def shap(self, X):
        logger.info("Computing SHAP values for %s", self.name)
        explainer = shap.TreeExplainer(self.model)
        return explainer.shap_values(X)

    def save(self):
        logger.info("Saving model %s", self.name)
        self.model.booster_.save_model(OUTPUT_PATH + "lightgbm.txt")
        logger.info("Saving done")

class LogRegModel(BaseModel):
    name = "log_reg"

    def _build_estimator(self, params):
        return LogisticRegression(**params)
    
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

    def shap(self, X):
        logger.info("Computing SHAP values for %s", self.name)
        explainer = shap.LinearExplainer(self.model, X)
        return explainer.shap_values(X)

    def save(self):
        logger.info("Saving model %s", self.name)
        joblib.dump(self.model, OUTPUT_PATH + "log_reg.joblib")
        logger.info("Saving done")


MODELS = {cls.name: cls for cls in (XGBoostModel, CatBoostModel, LightGBMModel, LogRegModel)}


def build_model(name, params=None):
    logger.info("Building model '%s' with params %s", name, params)
    return MODELS[name](params)
