from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression


def xgboost(params):
    return XGBClassifier(*params)


def catboost(params):
    return CatBoostClassifier(*params)


def lightgbm(params):
    return LGBMClassifier(*params)


def log_reg(params):
    return LogisticRegression(*params)
