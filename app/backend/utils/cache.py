import os
import json
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def _default_converter(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def load_from_cache(cache_path: str) -> Optional[dict]:
    if not os.path.exists(cache_path):
        logger.info("Cache miss: %s", cache_path)
        return None
    logger.info("Cache hit: %s", cache_path)
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning("Could not read cache file %s: %s", cache_path, e)
        return None


def save_to_cache(cache_path: str, data: dict) -> None:
    logger.info("Saving to cache: %s", cache_path)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        with open(cache_path, "w") as f:
            json.dump(data, f, default=_default_converter)
    except IOError as e:
        logger.error("Could not write cache file %s: %s", cache_path, e)
