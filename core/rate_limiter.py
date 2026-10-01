"""
FinCo AI - Rate Limiting

Protects the application against:
- API abuse
- Brute-force authentication
- Excessive Copilot requests
- Agent/tool abuse
- Expensive RAG operations
- Excessive ML/forecast requests
- Document upload abuse
- Denial-of-service style request bursts

The limiter uses a sliding-window algorithm and is safe to use
as a shared application service.

For a multi-instance production deployment, replace the in-memory
backend with Redis so limits are shared across all API instances.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Optional

from fastapi import HTTPException, Request, status

from app.core.exceptions import RateLimitExceededError


logger = logging.getLogger("finco.security")


# ============================================================
# Rate Limit Scope
# ============================================================

class RateLimitScope(str, Enum):
    IP = "ip"
    USER = "user"
    COMPANY = "company"
    GLOBAL = "global"


# ============================================================
# Rate Limit Configuration
# ============================================================

@dataclass(frozen=True)
class RateLimitRule:
    """
    Defines a rate limit.

    Example:
        60 requests / 60 seconds
    """

    max_requests: int
    window_seconds: int

    def __post_init__(self) -> None:
        if self.max_requests <= 0:
            raise ValueError(
                "max_requests must be greater than zero"
            )

        if self.window_seconds <= 0:
            raise ValueError(
                "window_seconds must be greater than zero"
            )


# ============================================================
# Default Application Rules
# ============================================================

DEFAULT_RULES: dict[str, RateLimitRule] = {

    # Authentication
    "auth_login": RateLimitRule(
        max_requests=5,
        window_seconds=60,
    ),

    "auth_register": RateLimitRule(
        max_requests=3,
        window_seconds=300,
    ),

    "auth_refresh": RateLimitRule(
        max_requests=20,
        window_seconds=60,
    ),

    # General API
    "api": RateLimitRule(
        max_requests=120,
        window_seconds=60,
    ),

    # Financial APIs
    "financial": RateLimitRule(
        max_requests=60,
        window_seconds=60,
    ),

    # Transactions
    "transactions": RateLimitRule(
        max_requests=100,
        window_seconds=60,
    ),

    # Document upload
    "document_upload": RateLimitRule(
        max_requests=20,
        window_seconds=300,
    ),

    # RAG
    "rag_query": RateLimitRule(
        max_requests=30,
        window_seconds=60,
    ),

    # Copilot
    "copilot": RateLimitRule(
        max_requests=30,
        window_seconds=60,
    ),

    # Agents
    "agent": RateLimitRule(
        max_requests=20,
        window_seconds=60,
    ),

    # Forecasting
    "forecast": RateLimitRule(
        max_requests=10,
        window_seconds=60,
    ),

    # Fraud detection
    "fraud": RateLimitRule(
        max_requests=20,
        window_seconds=60,
    ),

    # What-if simulations
    "what_if": RateLimitRule(
        max_requests=20,
        window_seconds=60,
    ),

    # Reports
    "reports": RateLimitRule(
        max_requests=20,
        window_seconds=60,
    ),

    # Export
    "export": RateLimitRule(
        max_requests=10,
        window_seconds=60,
    ),
}


# ============================================================
# Rate Limit Result
# ============================================================

@dataclass
class RateLimitResult:
    """
    Result returned by the rate limiter.
    """

    allowed: bool

    limit: int

    remaining: int

    reset_after: int

    retry_after: int = 0

    key: Optional[str] = None

    rule: Optional[str] = None


# ============================================================
# Sliding Window Store
# ============================================================

class InMemoryRateLimitStore:
    """
    Thread-safe in-memory sliding-window store.

    Suitable for:
    - local development
    - tests
    - single-process deployments

    Not sufficient for multiple API replicas because each
    process has its own memory.

    Production multi-instance deployment:
        Redis-backed implementation is recommended.
    """

    def __init__(self) -> None:

        self._requests: dict[
            str,
            deque[float],
        ] = defaultdict(deque)

        self._lock = Lock()

    def check(
        self,
        key: str,
        rule: RateLimitRule,
        now: Optional[float] = None,
    ) -> RateLimitResult:
        """
        Check and record a request using a sliding window.
        """

        current_time = (
            now
            if now is not None
            else time.monotonic()
        )

        window_start = (
            current_time
            - rule.window_seconds
        )

        with self._lock:

            timestamps = self._requests[key]

            # Remove expired requests.
            while (
                timestamps
                and timestamps[0] <= window_start
            ):
                timestamps.popleft()

            current_count = len(timestamps)

            if current_count >= rule.max_requests:

                oldest = timestamps[0]

                retry_after = max(
                    1,
                    int(
                        rule.window_seconds
                        - (
                            current_time
                            - oldest
                        )
                    ),
                )

                return RateLimitResult(
                    allowed=False,
                    limit=rule.max_requests,
                    remaining=0,
                    reset_after=retry_after,
                    retry_after=retry_after,
                    key=key,
                )

            timestamps.append(current_time)

            remaining = max(
                0,
                rule.max_requests
                - len(timestamps),
            )

            reset_after = (
                rule.window_seconds
            )

            if timestamps:
                reset_after = max(
                    1,
                    int(
                        rule.window_seconds
                        - (
                            current_time
                            - timestamps[0]
                        )
                    ),
                )

            return RateLimitResult(
                allowed=True,
                limit=rule.max_requests,
                remaining=remaining,
                reset_after=reset_after,
                retry_after=0,
                key=key,
            )

    def clear(
        self,
        key: Optional[str] = None,
    ) -> None:
        """Clear one key or the complete store."""

        with self._lock:

            if key is None:
                self._requests.clear()
            else:
                self._requests.pop(
                    key,
                    None,
                )

    def size(self) -> int:
        """Return the number of tracked keys."""

        with self._lock:
            return len(self._requests)


# ============================================================
# Rate Limiter
# ============================================================

class RateLimiter:
    """
    Application-wide rate limiter.

    Supports multiple scopes:
        IP
        USER
        COMPANY
        GLOBAL
    """

    def __init__(
        self,
        rules: Optional[
            dict[str, RateLimitRule]
        ] = None,
        store: Optional[
            InMemoryRateLimitStore
        ] = None,
    ) -> None:

        self.rules = (
            rules
            or DEFAULT_RULES.copy()
        )

        self.store = (
            store
            or InMemoryRateLimitStore()
        )

    # --------------------------------------------------------
    # Rule Management
    # --------------------------------------------------------

    def add_rule(
        self,
        name: str,
        max_requests: int,
        window_seconds: int,
    ) -> None:
        """Add or replace a rate-limit rule."""

        self.rules[name] = RateLimitRule(
            max_requests=max_requests,
            window_seconds=window_seconds,
        )

    def get_rule(
        self,
        name: str,
    ) -> RateLimitRule:
        """Retrieve a configured rule."""

        if name not in self.rules:
            raise ValueError(
                f"Unknown rate-limit rule: {name}"
            )

        return self.rules[name]

    # --------------------------------------------------------
    # Key Generation
    # --------------------------------------------------------

    @staticmethod
    def build_key(
        rule_name: str,
        scope: RateLimitScope,
        identifier: str,
    ) -> str:
        """
        Build a deterministic rate-limit key.
        """

        return (
            f"rate:{rule_name}:"
            f"{scope.value}:"
            f"{identifier}"
        )

    # --------------------------------------------------------
    # Core Check
    # --------------------------------------------------------

    def check(
        self,
        rule_name: str,
        scope: RateLimitScope,
        identifier: str,
    ) -> RateLimitResult:
        """
        Check whether a request is allowed.
        """

        rule = self.get_rule(
            rule_name
        )

        key = self.build_key(
            rule_name,
            scope,
            identifier,
        )

        result = self.store.check(
            key,
            rule,
        )

        result.rule = rule_name

        if not result.allowed:

            logger.warning(
                "Rate limit exceeded",
                extra={
                    "rate_limit_rule": rule_name,
                    "rate_limit_scope": scope.value,
                    "rate_limit_key": key,
                    "retry_after": result.retry_after,
                },
            )

        return result

    # --------------------------------------------------------
    # Require
    # --------------------------------------------------------

    def require(
        self,
        rule_name: str,
        scope: RateLimitScope,
        identifier: str,
    ) -> RateLimitResult:
        """
        Check the limit and raise when exceeded.
        """

        result = self.check(
            rule_name,
            scope,
            identifier,
        )

        if not result.allowed:

            raise RateLimitExceededError(
                message=(
                    "Rate limit exceeded. "
                    "Please retry later."
                ),
                details={
                    "rule": rule_name,
                    "scope": scope.value,
                    "limit": result.limit,
                    "retry_after": result.retry_after,
                },
            )

        return result

    # --------------------------------------------------------
    # Multi-Scope Check
    # --------------------------------------------------------

    def require_multiple(
        self,
        rule_name: str,
        identifiers: dict[
            RateLimitScope,
            str,
        ],
    ) -> RateLimitResult:
        """
        Apply the same rule to multiple scopes.

        Example:

            IP + user + company

        This prevents a user from bypassing limits simply
        by switching identifiers.
        """

        results: list[RateLimitResult] = []

        for scope, identifier in identifiers.items():

            result = self.require(
                rule_name,
                scope,
                identifier,
            )

            results.append(result)

        return min(
            results,
            key=lambda result: result.remaining,
        )

    # --------------------------------------------------------
    # Reset
    # --------------------------------------------------------

    def reset(
        self,
        rule_name: str,
        scope: RateLimitScope,
        identifier: str,
    ) -> None:

        key = self.build_key(
            rule_name,
            scope,
            identifier,
        )

        self.store.clear(key)


# ============================================================
# Global Limiter
# ============================================================

rate_limiter = RateLimiter()


# ============================================================
# Request Identity
# ============================================================

def get_client_ip(
    request: Request,
) -> str:
    """
    Get client IP.

    WARNING:
        Do not blindly trust X-Forwarded-For unless your
        reverse proxy is trusted and configured correctly.
    """

    # FastAPI/Starlette client information.
    if request.client:
        return request.client.host

    return "unknown"


def get_authenticated_user_id(
    request: Request,
) -> Optional[str]:
    """
    Retrieve authenticated user ID from request state.

    Your authentication middleware/dependency should set:

        request.state.user_id
    """

    return getattr(
        request.state,
        "user_id",
        None,
    )


def get_authenticated_company_id(
    request: Request,
) -> Optional[str]:
    """
    Retrieve authenticated company ID from request state.
    """

    return getattr(
        request.state,
        "company_id",
        None,
    )


# ============================================================
# FastAPI Dependency Factory
# ============================================================

def rate_limit_dependency(
    rule_name: str,
    scope: RateLimitScope = RateLimitScope.IP,
):
    """
    Create a FastAPI dependency for rate limiting.

    Example:

        login_rate_limit = rate_limit_dependency(
            "auth_login",
            RateLimitScope.IP,
        )

        @router.post(
            "/login",
            dependencies=[Depends(login_rate_limit)],
        )
        def login(...):
            ...
    """

    async def dependency(
        request: Request,
    ) -> None:

        identifier: Optional[str] = None

        if scope == RateLimitScope.IP:

            identifier = get_client_ip(
                request
            )

        elif scope == RateLimitScope.USER:

            identifier = (
                get_authenticated_user_id(
                    request
                )
            )

            if identifier is None:
                identifier = get_client_ip(
                    request
                )

        elif scope == RateLimitScope.COMPANY:

            identifier = (
                get_authenticated_company_id(
                    request
                )
            )

            if identifier is None:
                identifier = get_client_ip(
                    request
                )

        elif scope == RateLimitScope.GLOBAL:

            identifier = "global"

        if not identifier:
            identifier = "unknown"

        result = rate_limiter.check(
            rule_name,
            scope,
            str(identifier),
        )

        # Expose rate-limit information.
        request.state.rate_limit = result

        if not result.allowed:

            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "rate_limit_exceeded",
                    "message": (
                        "Too many requests. "
                        "Please retry later."
                    ),
                    "retry_after": result.retry_after,
                },
                headers={
                    "Retry-After": str(
                        result.retry_after
                    ),
                    "X-RateLimit-Limit": str(
                        result.limit
                    ),
                    "X-RateLimit-Remaining": "0",
                },
            )

    return dependency


# ============================================================
# Specialized Dependencies
# ============================================================

auth_rate_limit = rate_limit_dependency(
    "auth_login",
    RateLimitScope.IP,
)


api_rate_limit = rate_limit_dependency(
    "api",
    RateLimitScope.USER,
)


financial_rate_limit = rate_limit_dependency(
    "financial",
    RateLimitScope.COMPANY,
)


transaction_rate_limit = rate_limit_dependency(
    "transactions",
    RateLimitScope.USER,
)


document_upload_rate_limit = rate_limit_dependency(
    "document_upload",
    RateLimitScope.USER,
)


rag_rate_limit = rate_limit_dependency(
    "rag_query",
    RateLimitScope.USER,
)


copilot_rate_limit = rate_limit_dependency(
    "copilot",
    RateLimitScope.USER,
)


agent_rate_limit = rate_limit_dependency(
    "agent",
    RateLimitScope.USER,
)


forecast_rate_limit = rate_limit_dependency(
    "forecast",
    RateLimitScope.COMPANY,
)


fraud_rate_limit = rate_limit_dependency(
    "fraud",
    RateLimitScope.COMPANY,
)


what_if_rate_limit = rate_limit_dependency(
    "what_if",
    RateLimitScope.USER,
)


report_rate_limit = rate_limit_dependency(
    "reports",
    RateLimitScope.USER,
)


export_rate_limit = rate_limit_dependency(
    "export",
    RateLimitScope.USER,
)


# ============================================================
# Rate Limit Headers
# ============================================================

def get_rate_limit_headers(
    result: RateLimitResult,
) -> dict[str, str]:
    """
    Generate standard rate-limit headers.
    """

    headers = {
        "X-RateLimit-Limit": str(
            result.limit
        ),

        "X-RateLimit-Remaining": str(
            result.remaining
        ),

        "X-RateLimit-Reset": str(
            result.reset_after
        ),
    }

    if result.retry_after:
        headers["Retry-After"] = str(
            result.retry_after
        )

    return headers


# ============================================================
# Cleanup
# ============================================================

def clear_rate_limits() -> None:
    """
    Clear all in-memory rate-limit state.

    Useful for:
    - tests
    - development
    - administrative reset
    """

    rate_limiter.store.clear()


def get_rate_limit_store_size() -> int:
    """Return number of active rate-limit keys."""

    return rate_limiter.store.size()