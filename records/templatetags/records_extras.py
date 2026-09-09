"""Small template helpers."""
from django import template

register = template.Library()


@register.filter
def get_item(mapping, key):
    if isinstance(mapping, dict):
        return mapping.get(key, "")
    return ""


@register.filter
def pct(value, digits=1):
    try:
        return f"{float(value) * 100:.{digits}f}"
    except (TypeError, ValueError):
        return "0"


@register.filter
def conf_class(value):
    """CSS class for a 0-1 confidence score."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = 0
    if v >= 0.85:
        return "ok"
    if v >= 0.60:
        return "warn"
    return "bad"


@register.filter
def status_class(status):
    return {
        "PROCESSED": "ok", "VERIFIED": "ok", "UPLOADED": "info",
        "PROCESSING": "info", "PENDING": "warn", "FAILED": "bad",
        "REJECTED": "bad",
    }.get(status, "info")
