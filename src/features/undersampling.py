from imblearn.under_sampling import RandomUnderSampler
import logging

logger = logging.getLogger(__name__)


def undersampling(X_train, y_train):
    rs = RandomUnderSampler(random_state=42)

    logger.info(
        "Start undersampling X_train shape: %s, y_train_shape: %s", X_train.shape, y_train.shape
    )
    X_train_und, y_train_und = rs.fit_resample(X_train, y_train)
    logger.info(
        "Finished undersampling X_train_und shape: %s, y_train_und shape: %s",
        X_train_und.shape,
        y_train_und.shape,
    )

    return X_train_und, y_train_und
