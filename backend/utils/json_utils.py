import math
from typing import Any


def sanitize_for_json(data: Any) -> Any:
    """Recursively replace NaN, Infinity, and -Infinity with None for JSON compatibility."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    else:
        return data
