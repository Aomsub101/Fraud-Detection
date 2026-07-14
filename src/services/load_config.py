import logging

logger = logging.getLogger(__name__)


def load_config(path="config.txt"):
    logger.info("Loading config from %s", path)
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            config[key.strip()] = value.strip()
    logger.info("Loaded config: %s", config)
    return config
