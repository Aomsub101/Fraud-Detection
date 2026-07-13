FE_HANDLED_COLS = [
    "prev_address_months_count",
    "bank_months_count",
    "device_distinct_emails_8w",
]


def features_engineer(df):
    df_feat = df.copy()

    for col in FE_HANDLED_COLS:
        df_feat[f"is_{col}_available"] = df_feat[col].notna().astype(int)
        df_feat.drop(columns=[col], inplace=True)

    return df_feat
