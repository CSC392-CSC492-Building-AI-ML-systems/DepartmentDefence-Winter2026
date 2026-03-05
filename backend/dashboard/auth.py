import os
from functools import wraps


def _is_dashboard_enabled() -> bool:
    secret = os.getenv("DASHBOARD_ACCESS_KEY", "").strip()
    if not secret:
        return False
    if secret.lower() in {"0", "false", "off", "disabled"}:
        return False
    return True


def require_dashboard_secret(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        if not _is_dashboard_enabled():
            return ("", 404)
        return fn(*args, **kwargs)

    return wrapper
