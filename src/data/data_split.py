TARGET = "fraud_bool"


def data_split(df):
    train_df = df[df["month"] <= 5]
    test_df = df[df["month"] >= 6]

    feature_cols = [c for c in df.columns if c not in (TARGET, "month")]

    X_train = train_df[feature_cols]
    y_train = train_df[TARGET]
    X_test = test_df[feature_cols]
    y_test = test_df[TARGET]

    return X_train, X_test, y_train, y_test
