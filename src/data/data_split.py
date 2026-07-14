import logging

logger = logging.getLogger(__name__)

TARGET = "fraud_bool"


def data_split(df):
    logger.info("Splitting by month (train: 0-5, test: 6-7)")
    train_df = df[df["month"] <= 5]
    test_df = df[df["month"] >= 6]

    feature_cols = [c for c in df.columns if c not in (TARGET, "month")]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    logger.info("Split done: X_train=%s, X_test=%s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test
