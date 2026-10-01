"""
backend/app/api/health.py

Health and readiness endpoints for FinCo AI.

Endpoints:
    GET /health
    GET /health/live
    GET /health/ready
    GET /health/dependencies

Design goals:
    - Fast liveness checks
    - Dependency-aware readiness checks
    - Database connectivity validation
    - Vector-store connectivity validation
    - Safe error reporting
    - No authentication required for health endpoints
    - Suitable for Docker/Kubernetes/load balancers
    - Avoid exposing secrets or sensitive infrastructure details
"""

from __future__ import annotations

import asyncio
import logging
import os
import platform
import socket
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from fastapi import APIRouter, Response, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/health",
    tags=["Health"],
)


# ============================================================================
# ENUMS
# ============================================================================


class HealthStatus(str, Enum):
    """Overall health state."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyStatus(str, Enum):
    """Dependency health state."""

    UP = "up"
    DOWN = "down"
    DEGRADED = "degraded"
    NOT_CONFIGURED = "not_configured"


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class ServiceInfo(BaseModel):
    """Basic application/service metadata."""

    name: str
    version: str
    environment: str
    hostname: str


class DependencyHealth(BaseModel):
    """Health information for an external/internal dependency."""

    name: str
    status: DependencyStatus
    latency_ms: float | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    """General health response."""

    status: HealthStatus
    service: ServiceInfo
    timestamp: datetime
    uptime_seconds: float
    dependencies: list[DependencyHealth] = Field(default_factory=list)


class LivenessResponse(BaseModel):
    """Kubernetes/container liveness response."""

    status: str
    timestamp: datetime


class ReadinessResponse(BaseModel):
    """Readiness response used by load balancers/orchestrators."""

    status: HealthStatus
    ready: bool
    timestamp: datetime
    dependencies: list[DependencyHealth] = Field(default_factory=list)


class DependencyHealthResponse(BaseModel):
    """Detailed dependency health response."""

    status: HealthStatus
    timestamp: datetime
    dependencies: list[DependencyHealth] = Field(default_factory=list)


# ============================================================================
# APPLICATION METADATA
# ============================================================================


_APP_START_TIME = time.monotonic()


def _utc_now() -> datetime:
    """Return timezone-aware UTC timestamp."""

    return datetime.now(timezone.utc)


def _get_app_name() -> str:
    """Return application name without exposing infrastructure details."""

    return os.getenv("APP_NAME", "FinCo AI")


def _get_app_version() -> str:
    """Return application version."""

    return os.getenv("APP_VERSION", "1.0.0")


def _get_environment() -> str:
    """
    Return sanitized environment name.

    Never expose credentials, connection strings, API keys,
    or other sensitive environment variables.
    """

    return os.getenv("APP_ENV", "development")


def _get_hostname() -> str:
    """Return current service hostname."""

    try:
        return socket.gethostname()
    except Exception:
        return "unknown"


def _service_info() -> ServiceInfo:
    """Build service metadata."""

    return ServiceInfo(
        name=_get_app_name(),
        version=_get_app_version(),
        environment=_get_environment(),
        hostname=_get_hostname(),
    )


def _uptime_seconds() -> float:
    """Return application uptime."""

    return round(time.monotonic() - _APP_START_TIME, 3)


# ============================================================================
# DATABASE HEALTH
# ============================================================================


async def check_database() -> DependencyHealth:
    """
    Check database connectivity.

    Expected architecture:
        backend.app.database.session
        backend.app.database.connection

    The check attempts to obtain an async SQLAlchemy session and
    execute SELECT 1.

    This function intentionally imports the DB layer lazily so that
    importing the health router does not initialize the database.
    """

    dependency_name = "database"
    started = time.perf_counter()

    try:
        from sqlalchemy import text

        # Prefer the project's session dependency/factory.
        #
        # This supports common FinCo AI layouts where session.py
        # exposes AsyncSessionLocal/session_factory.
        from backend.app.database.session import AsyncSessionLocal

        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.UP,
            latency_ms=latency_ms,
            message="Database connection is healthy.",
        )

    except ImportError as exc:
        logger.warning(
            "Database health check is unavailable: %s",
            exc,
        )

        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.NOT_CONFIGURED,
            message="Database session is not configured.",
        )

    except Exception as exc:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        logger.exception("Database health check failed.")

        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.DOWN,
            latency_ms=latency_ms,
            message="Database connectivity check failed.",
        )


# ============================================================================
# VECTOR STORE HEALTH
# ============================================================================


async def check_vector_store() -> DependencyHealth:
    """
    Check Qdrant/vector-store connectivity.

    Uses the project's VectorStore abstraction when available.

    The implementation intentionally avoids returning:
        - API keys
        - connection strings
        - hostnames containing credentials
        - raw exception messages

    This keeps public health endpoints safe.
    """

    dependency_name = "vector_store"
    started = time.perf_counter()

    try:
        from backend.app.rag.vector_store import VectorStore

        vector_store = VectorStore()

        # The project VectorStore should expose an async health method.
        health_method = getattr(vector_store, "health", None)

        if health_method is None:
            return DependencyHealth(
                name=dependency_name,
                status=DependencyStatus.NOT_CONFIGURED,
                message="Vector store health interface is unavailable.",
            )

        result = health_method()

        if asyncio.iscoroutine(result):
            result = await result

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        # Support both boolean and dictionary-style health responses.
        if isinstance(result, bool):
            healthy = result
        elif isinstance(result, dict):
            healthy = bool(
                result.get("healthy")
                or result.get("status") in {"healthy", "ok", "up"}
            )
        else:
            healthy = True

        return DependencyHealth(
            name=dependency_name,
            status=(
                DependencyStatus.UP
                if healthy
                else DependencyStatus.DEGRADED
            ),
            latency_ms=latency_ms,
            message=(
                "Vector store connection is healthy."
                if healthy
                else "Vector store is reachable but reported an unhealthy state."
            ),
        )

    except ImportError as exc:
        logger.warning(
            "Vector-store health check is unavailable: %s",
            exc,
        )

        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.NOT_CONFIGURED,
            message="Vector store is not configured.",
        )

    except Exception:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        logger.exception("Vector-store health check failed.")

        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.DOWN,
            latency_ms=latency_ms,
            message="Vector-store connectivity check failed.",
        )


# ============================================================================
# REDIS HEALTH
# ============================================================================


async def check_redis() -> DependencyHealth:
    """
    Check Redis connectivity when Redis is configured.

    FinCo AI may use Redis for:
        - caching
        - rate limiting
        - background jobs
        - distributed locks
        - session/state management

    Redis is treated as optional here. If REDIS_URL is not configured,
    the dependency is reported as NOT_CONFIGURED rather than unhealthy.
    """

    dependency_name = "redis"
    started = time.perf_counter()

    redis_url = os.getenv("REDIS_URL")

    if not redis_url:
        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.NOT_CONFIGURED,
            message="Redis is not configured.",
        )

    try:
        import redis.asyncio as redis

        client = redis.from_url(
            redis_url,
            decode_responses=True,
        )

        try:
            result = await client.ping()
        finally:
            await client.aclose()

        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        if result is True:
            return DependencyHealth(
                name=dependency_name,
                status=DependencyStatus.UP,
                latency_ms=latency_ms,
                message="Redis connection is healthy.",
            )

        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.DEGRADED,
            latency_ms=latency_ms,
            message="Redis responded unexpectedly.",
        )

    except ImportError:
        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.NOT_CONFIGURED,
            message="Redis client is not installed.",
        )

    except Exception:
        latency_ms = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        logger.exception("Redis health check failed.")

        return DependencyHealth(
            name=dependency_name,
            status=DependencyStatus.DOWN,
            latency_ms=latency_ms,
            message="Redis connectivity check failed.",
        )


# ============================================================================
# DEPENDENCY ORCHESTRATION
# ============================================================================


async def run_dependency_checks() -> list[DependencyHealth]:
    """
    Execute dependency health checks concurrently.

    Concurrent checks prevent a slow dependency from unnecessarily
    delaying unrelated checks.
    """

    checks = await asyncio.gather(
        check_database(),
        check_vector_store(),
        check_redis(),
        return_exceptions=True,
    )

    dependencies: list[DependencyHealth] = []

    for result in checks:
        if isinstance(result, DependencyHealth):
            dependencies.append(result)
            continue

        logger.exception(
            "Unexpected exception during dependency health check.",
            exc_info=result,
        )

        dependencies.append(
            DependencyHealth(
                name="unknown",
                status=DependencyStatus.DOWN,
                message="Dependency health check failed.",
            )
        )

    return dependencies


def determine_overall_status(
    dependencies: list[DependencyHealth],
) -> HealthStatus:
    """
    Determine overall health from dependency states.

    Rules:
        - Any DOWN dependency -> UNHEALTHY
        - Any DEGRADED dependency -> DEGRADED
        - Configured dependencies all UP -> HEALTHY
        - NOT_CONFIGURED dependencies are ignored
    """

    states = {dependency.status for dependency in dependencies}

    if DependencyStatus.DOWN in states:
        return HealthStatus.UNHEALTHY

    if DependencyStatus.DEGRADED in states:
        return HealthStatus.DEGRADED

    return HealthStatus.HEALTHY


def readiness_success(
    dependencies: list[DependencyHealth],
) -> bool:
    """
    Determine whether the application can receive traffic.

    Required dependencies:
        - database

    Optional dependencies:
        - vector store
        - redis

    This distinction prevents optional infrastructure from taking
    down the entire application.
    """

    required_dependencies = {"database"}

    for dependency in dependencies:
        if dependency.name not in required_dependencies:
            continue

        if dependency.status != DependencyStatus.UP:
            return False

    return True


# ============================================================================
# PUBLIC ENDPOINTS
# ============================================================================


@router.get(
    "",
    response_model=HealthResponse,
    summary="Application health",
    description=(
        "Returns overall application health and dependency status."
    ),
)
async def health(response: Response) -> HealthResponse:
    """
    Comprehensive health endpoint.

    Suitable for:
        - monitoring dashboards
        - operations
        - manual diagnostics

    It may perform dependency checks and therefore should not be used
    as a high-frequency liveness probe.
    """

    dependencies = await run_dependency_checks()

    overall_status = determine_overall_status(dependencies)

    if overall_status == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    elif overall_status == HealthStatus.DEGRADED:
        response.status_code = status.HTTP_200_OK

    return HealthResponse(
        status=overall_status,
        service=_service_info(),
        timestamp=_utc_now(),
        uptime_seconds=_uptime_seconds(),
        dependencies=dependencies,
    )


@router.get(
    "/live",
    response_model=LivenessResponse,
    summary="Liveness probe",
    description=(
        "Lightweight process liveness check. "
        "Does not contact external dependencies."
    ),
)
async def liveness() -> LivenessResponse:
    """
    Kubernetes/container liveness endpoint.

    If this endpoint responds successfully, the Python process
    and FastAPI application are alive.

    IMPORTANT:
        Do not perform database or network calls here.
    """

    return LivenessResponse(
        status="alive",
        timestamp=_utc_now(),
    )


@router.get(
    "/ready",
    response_model=ReadinessResponse,
    summary="Readiness probe",
    description=(
        "Checks whether required dependencies are available "
        "and the service is ready to receive traffic."
    ),
)
async def readiness(response: Response) -> ReadinessResponse:
    """
    Kubernetes/load-balancer readiness endpoint.

    Unlike liveness, this verifies required dependencies.
    """

    dependencies = await run_dependency_checks()

    ready = readiness_success(dependencies)

    overall_status = (
        HealthStatus.HEALTHY
        if ready
        else HealthStatus.UNHEALTHY
    )

    if not ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(
        status=overall_status,
        ready=ready,
        timestamp=_utc_now(),
        dependencies=dependencies,
    )


@router.get(
    "/dependencies",
    response_model=DependencyHealthResponse,
    summary="Dependency health",
    description="Returns detailed health status for service dependencies.",
)
async def dependency_health(
    response: Response,
) -> DependencyHealthResponse:
    """
    Dependency diagnostics endpoint.

    Useful for:
        - operational troubleshooting
        - observability
        - deployment verification
    """

    dependencies = await run_dependency_checks()

    overall_status = determine_overall_status(dependencies)

    if overall_status == HealthStatus.UNHEALTHY:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return DependencyHealthResponse(
        status=overall_status,
        timestamp=_utc_now(),
        dependencies=dependencies,
    )


# ============================================================================
# OPTIONAL SYSTEM INFORMATION
# ============================================================================


@router.get(
    "/info",
    summary="Service information",
    description="Returns non-sensitive application runtime information.",
)
async def service_info() -> dict[str, Any]:
    """
    Return safe operational metadata.

    Deliberately excludes:
        - environment variables
        - API keys
        - database URLs
        - Qdrant URLs
        - Redis URLs
        - credentials
    """

    return {
        "service": _service_info().model_dump(),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.system(),
            "architecture": platform.machine(),
        },
        "uptime_seconds": _uptime_seconds(),
        "timestamp": _utc_now(),
    }