import logging

logger = logging.getLogger(__name__)

FE_HANDLED_COLS = [
    "prev_address_months_count",
    "bank_months_count",
    "device_distinct_emails_8w",
]


def features_engineer(df):
    logger.info("Feature engineering: availability flags for %s", FE_HANDLED_COLS)
    df_feat = df.copy()

    for col in FE_HANDLED_COLS:
        df_feat[f"is_{col}_available"] = df_feat[col].notna().astype(int)

    df_feat = df_feat.drop(columns=FE_HANDLED_COLS)

    logger.info("Feature engineering complete: shape %s", df_feat.shape)
    return df_feat
