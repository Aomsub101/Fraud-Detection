import json
import logging

logger = logging.getLogger(__name__)


def load_config(path="config.json"):
    logger.info("Loading config from %s", path)
    with open(path) as f:
        config = json.load(f)
    logger.info("Loaded config: model=%s", config.get("model"))
    return config
