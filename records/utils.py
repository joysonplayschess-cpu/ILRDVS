"""Shared helpers: audit trail + role-based access control."""
from functools import wraps

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect

from .models import AuditLog


def client_ip(request):
    xff = request.META.get("HTTP_X_FORWARDED_FOR")
    if xff:
        return xff.split(",")[0].strip()
    return request.META.get("REMOTE_ADDR")


def log_audit(user, action, obj=None, detail="", request=None):
    """Write an immutable audit-trail entry."""
    return AuditLog.objects.create(
        user=user if getattr(user, "is_authenticated", False) else None,
        user_label=(user.get_username() if getattr(user, "is_authenticated", False)
                    else "system"),
        action=action,
        object_type=obj.__class__.__name__ if obj is not None else "",
        object_id=str(getattr(obj, "pk", "") or ""),
        detail=detail[:2000],
        ip_address=client_ip(request) if request is not None else None,
    )


def get_role(user):
    if not user.is_authenticated:
        return None
    if user.is_superuser:
        return "ADMIN"
    profile = getattr(user, "profile", None)
    return profile.role if profile else "VIEWER"


def role_required(*roles):
    """Restrict a view to the given roles; friendly redirect otherwise."""
    def decorator(view_func):
        @wraps(view_func)
        @login_required
        def wrapper(request, *args, **kwargs):
            role = get_role(request.user)
            if role in roles:
                return view_func(request, *args, **kwargs)
            messages.warning(
                request,
                f"Access denied: this section requires the "
                f"{'/'.join(roles)} role (you are logged in as {role}).")
            return redirect("dashboard")
        return wrapper
    return decorator
