from src.data.load_data import load_data
from src.data.clean_data import clean_data
from src.data.preprocess import preprocess
from src.data.data_split import data_split
from src.features.features_engineer import features_engineer
from src.models.models import xgboost
from src.services.metrics import evaluate_model
from src.services.load_config import load_config


def main():
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

    # model
    params = {
        "xgboost": {},
        "catboost": {},
        "lightgbm": {},
    }

    model_name = config["model"]
    model = xgboost(params=params[model_name])

    # train
    model.fit(X_train, y_train)

    # predict
    y_pred = model.predict(X_test)

    # evaluate
    evaluate_model(y_test, y_pred)


if __name__ == "__main__":
    main()
