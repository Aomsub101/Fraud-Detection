import logging

logger = logging.getLogger(__name__)

DROP_COLS = ["device_fraud_count"]


def clean_data(df):
    logger.info("Cleaning data: dropping columns %s", DROP_COLS)
    df_clean = df.copy()
    df_clean.drop(columns=DROP_COLS, inplace=True)
    logger.info("Cleaned data shape: %s", df_clean.shape)
    return df_clean
