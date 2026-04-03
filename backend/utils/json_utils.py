import math
from typing import Any


def sanitize_for_json(data: Any) -> Any:
    """Recursively replace NaN/Infinity with None and convert non-serializable objects (like Timestamps) to strings."""
    if isinstance(data, dict):
        return {k: sanitize_for_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_for_json(v) for v in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif hasattr(data, "isoformat"):
        return data.isoformat()
    elif hasattr(data, "item"):  # Handle numpy scalars
        try:
            return sanitize_for_json(data.item())
        except Exception:
            return str(data)
    elif "Timestamp" in str(type(data)):  # Final fallback for pandas Timestamps
        try:
            return data.isoformat()
        except Exception:
            return str(data)
    else:
        return data
