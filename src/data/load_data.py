# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
import logging
import os
from kagglehub import KaggleDatasetAdapter
import pandas as pd

logger = logging.getLogger(__name__)


def load_data():
    # Name of the file within the kaggle dataset.
    file_path = "Base.csv"
    # Local cached copy (must match where the file is saved below).
    local_path = "data/Base.csv"

    # Use the cached copy if it exists.
    if os.path.exists(local_path):
        logger.info("File '%s' already exists. Loading cached data ...", local_path)
        df = pd.read_csv(local_path)
        logger.info("Loaded cached data: %d rows, %d columns", df.shape[0], df.shape[1])
        return df

    # Load the latest version
    logger.info("Loading '%s' from kagglehub...", file_path)
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "sgpjesus/bank-account-fraud-dataset-neurips-2022",
        file_path,
        # Provide any additional arguments like
        # sql_query or pandas_kwargs. See the
        # documenation for more information:
        # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
    logger.info("Loaded data: %d rows, %d columns", df.shape[0], df.shape[1])
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    df.to_csv(local_path, index=False)
    logger.info("Saved raw data to %s", local_path)

    return df
