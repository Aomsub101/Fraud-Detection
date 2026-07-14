DROP_COLS = ['device_fraud_count']


def clean_data(df):
    df_clean = df.copy()
    df_clean.drop(columns=DROP_COLS, inplace=True)
    return df_clean
