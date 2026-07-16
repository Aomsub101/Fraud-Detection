from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.preprocess import preprocess
from src.data.data_split import data_split
from src.features.features_engineer import features_engineer
from src.features.undersampling import undersampling
from src.models.models import build_model
from src.services.metrics import evaluate_model
from src.services.load_config import load_config
from src.services.output_analysis import output_analysis
from src.services.logging_config import setup_logging
import logging
import matplotlib

logger = logging.getLogger(__name__)


def main():
    # Use a non-interactive backend so plots are saved to files without opening
    # windows or clearing the figure (which had produced blank saved images).
    matplotlib.use("Agg")

    setup_logging()
    logger.info("=== Fraud detection pipeline started ===")

    # load config
    config = load_config()

    # load data
    df = load_data()

    # clean data
    df_clean = clean_data(df)

    # preprocess data
    df_prep = preprocess(df_clean)

    # feature engineering
    df_feat = features_engineer(df_prep)

    # data split
    X_train, X_test, y_train, y_test = data_split(df_feat)

    # undersampling
    X_train_und, y_train_und = undersampling(X_train, y_train)

    # model
    param_keys = {
        "xgboost": "XGBoost_best_params",
        "catboost": "CatBoost_best_params",
        "lightgbm": "LightGBM_best_params",
        "log_reg": "LogReg_best_params",
    }

    model_name = config["model"]
    params = {**config.get(param_keys[model_name], {})}
    model = build_model(name=model_name, params=params)

    # train
    model.fit(X=X_train_und, y=y_train_und)

    # predict
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)

    # evaluate
    metrics = evaluate_model(y_test, y_pred, y_score)

    output_analysis(model, metrics=metrics, X_test=X_test, y_test=y_test, y_score=y_score)

    model.save()

    logger.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
