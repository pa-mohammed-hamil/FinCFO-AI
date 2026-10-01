"""
FinCo AI - Centralized Logging

Provides:
- Application-wide logging configuration
- Console and file logging
- Structured log formatting
- Request ID / correlation ID support
- Environment-aware log levels
- Separate audit/security loggers
- Third-party logger noise reduction

Usage:
    from app.core.logging import get_logger

    logger = get_logger(__name__)

    logger.info("Transaction created", extra={
        "company_id": company_id,
        "transaction_id": transaction_id,
    })
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional


# ============================================================
# Request Context
# ============================================================

_request_id: ContextVar[Optional[str]] = ContextVar(
    "request_id",
    default=None,
)

_user_id: ContextVar[Optional[str]] = ContextVar(
    "user_id",
    default=None,
)

_company_id: ContextVar[Optional[str]] = ContextVar(
    "company_id",
    default=None,
)


def generate_request_id() -> str:
    """Generate a unique request/correlation ID."""
    return str(uuid.uuid4())


def set_request_context(
    request_id: Optional[str] = None,
    user_id: Optional[str] = None,
    company_id: Optional[str] = None,
) -> str:
    """
    Set request-scoped logging context.

    Returns:
        The active request ID.
    """
    active_request_id = request_id or generate_request_id()

    _request_id.set(active_request_id)
    _user_id.set(user_id)
    _company_id.set(company_id)

    return active_request_id


def clear_request_context() -> None:
    """Clear request-scoped logging context."""
    _request_id.set(None)
    _user_id.set(None)
    _company_id.set(None)


def get_request_id() -> Optional[str]:
    return _request_id.get()


def get_user_id() -> Optional[str]:
    return _user_id.get()


def get_company_id() -> Optional[str]:
    return _company_id.get()


# ============================================================
# Configuration
# ============================================================

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()

LOG_DIR = Path(
    os.getenv(
        "LOG_DIR",
        "logs",
    )
)

LOG_FILE = os.getenv(
    "LOG_FILE",
    "finco.log",
)

MAX_LOG_FILE_SIZE = int(
    os.getenv(
        "MAX_LOG_FILE_SIZE",
        str(10 * 1024 * 1024),  # 10 MB
    )
)

BACKUP_COUNT = int(
    os.getenv(
        "LOG_BACKUP_COUNT",
        "5",
    )
)

ENABLE_FILE_LOGGING = (
    os.getenv(
        "ENABLE_FILE_LOGGING",
        "true",
    ).lower()
    in {"true", "1", "yes"}
)


# ============================================================
# Sensitive Data Protection
# ============================================================

SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "authorization",
    "cookie",
    "session",
    "credit_card",
    "card_number",
    "cvv",
    "bank_account",
}


def _sanitize_value(
    value: Any,
    key: Optional[str] = None,
) -> Any:
    """
    Remove sensitive values before they reach logs.
    """

    if key and key.lower() in SENSITIVE_KEYS:
        return "[REDACTED]"

    if isinstance(value, dict):
        return {
            str(k): _sanitize_value(v, str(k))
            for k, v in value.items()
        }

    if isinstance(value, list):
        return [
            _sanitize_value(item)
            for item in value
        ]

    if isinstance(value, tuple):
        return tuple(
            _sanitize_value(item)
            for item in value
        )

    return value


# ============================================================
# Structured Formatter
# ============================================================

class StructuredFormatter(logging.Formatter):
    """
    JSON-friendly structured formatter.

    Example:
    {
        "timestamp": "...",
        "level": "INFO",
        "logger": "app.api.transactions",
        "message": "Transaction created",
        "request_id": "...",
        "company_id": "...",
        "user_id": "..."
    }
    """

    RESERVED_ATTRIBUTES = {
        "name",
        "msg",
        "args",
        "levelname",
        "levelno",
        "pathname",
        "filename",
        "module",
        "exc_info",
        "exc_text",
        "stack_info",
        "lineno",
        "funcName",
        "created",
        "msecs",
        "relativeCreated",
        "thread",
        "threadName",
        "processName",
        "process",
        "taskName",
    }

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        payload: dict[str, Any] = {
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),

            "level": record.levelname,

            "logger": record.name,

            "message": record.getMessage(),

            "environment": ENVIRONMENT,

            "request_id": get_request_id(),

            "user_id": get_user_id(),

            "company_id": get_company_id(),
        }

        # Add custom `extra={...}` fields.
        for key, value in record.__dict__.items():

            if key in self.RESERVED_ATTRIBUTES:
                continue

            if key.startswith("_"):
                continue

            payload[key] = _sanitize_value(
                value,
                key,
            )

        # Exception information.
        if record.exc_info:

            payload["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": "".join(
                    traceback.format_exception(
                        *record.exc_info
                    )
                ),
            }

        return json.dumps(
            _sanitize_value(payload),
            default=str,
            ensure_ascii=False,
        )


# ============================================================
# Development Formatter
# ============================================================

class ConsoleFormatter(logging.Formatter):
    """
    Human-readable formatter for local development.
    """

    def format(
        self,
        record: logging.LogRecord,
    ) -> str:

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y-%m-%d %H:%M:%S")

        request_id = get_request_id()

        request_part = (
            f" request_id={request_id}"
            if request_id
            else ""
        )

        company_id = get_company_id()

        company_part = (
            f" company_id={company_id}"
            if company_id
            else ""
        )

        message = (
            f"{timestamp} | "
            f"{record.levelname:<8} | "
            f"{record.name} | "
            f"{record.getMessage()}"
            f"{request_part}"
            f"{company_part}"
        )

        if record.exc_info:
            message += "\n" + "".join(
                traceback.format_exception(
                    *record.exc_info
                )
            )

        return message


# ============================================================
# Filters
# ============================================================

class ContextFilter(logging.Filter):
    """
    Adds request context to LogRecord.

    This makes request metadata available to
    formatters and logging handlers.
    """

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:

        record.request_id = get_request_id()
        record.user_id = get_user_id()
        record.company_id = get_company_id()

        return True


class HealthCheckFilter(logging.Filter):
    """
    Prevents health-check requests from flooding logs.

    Can be disabled with:
        LOG_HEALTH_CHECKS=true
    """

    def filter(
        self,
        record: logging.LogRecord,
    ) -> bool:

        if (
            os.getenv(
                "LOG_HEALTH_CHECKS",
                "false",
            ).lower()
            in {"true", "1", "yes"}
        ):
            return True

        message = record.getMessage().lower()

        health_patterns = (
            "/health",
            "health check",
            "readiness",
            "liveness",
        )

        return not any(
            pattern in message
            for pattern in health_patterns
        )


# ============================================================
# Logger Factory
# ============================================================

def get_logger(
    name: Optional[str] = None,
) -> logging.Logger:
    """
    Return a configured application logger.

    Example:
        logger = get_logger(__name__)
    """

    return logging.getLogger(
        name or "finco"
    )


# ============================================================
# Logging Configuration
# ============================================================

_configured = False


def configure_logging() -> None:
    """
    Configure application-wide logging.

    Safe to call multiple times.
    """

    global _configured

    if _configured:
        return

    root_logger = logging.getLogger()

    root_logger.setLevel(
        getattr(
            logging,
            LOG_LEVEL,
            logging.INFO,
        )
    )

    # Prevent duplicate handlers when using
    # uvicorn reload / tests.
    root_logger.handlers.clear()

    context_filter = ContextFilter()

    health_filter = HealthCheckFilter()

    # --------------------------------------------------------
    # Console Handler
    # --------------------------------------------------------

    console_handler = logging.StreamHandler(
        sys.stdout
    )

    console_handler.setLevel(
        getattr(
            logging,
            LOG_LEVEL,
            logging.INFO,
        )
    )

    if ENVIRONMENT in {
        "production",
        "prod",
    }:

        console_handler.setFormatter(
            StructuredFormatter()
        )

    else:

        console_handler.setFormatter(
            ConsoleFormatter()
        )

    console_handler.addFilter(
        context_filter
    )

    console_handler.addFilter(
        health_filter
    )

    root_logger.addHandler(
        console_handler
    )

    # --------------------------------------------------------
    # File Handler
    # --------------------------------------------------------

    if ENABLE_FILE_LOGGING:

        LOG_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        log_path = LOG_DIR / LOG_FILE

        file_handler = RotatingFileHandler(
            filename=log_path,
            maxBytes=MAX_LOG_FILE_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )

        file_handler.setLevel(
            getattr(
                logging,
                LOG_LEVEL,
                logging.INFO,
            )
        )

        file_handler.setFormatter(
            StructuredFormatter()
        )

        file_handler.addFilter(
            context_filter
        )

        file_handler.addFilter(
            health_filter
        )

        root_logger.addHandler(
            file_handler
        )

    # --------------------------------------------------------
    # Third-party Logger Levels
    # --------------------------------------------------------

    logging.getLogger(
        "uvicorn.access"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "uvicorn.error"
    ).setLevel(logging.INFO)

    logging.getLogger(
        "sqlalchemy.engine"
    ).setLevel(
        logging.WARNING
    )

    logging.getLogger(
        "httpx"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "httpcore"
    ).setLevel(logging.WARNING)

    logging.getLogger(
        "multipart"
    ).setLevel(logging.WARNING)

    # --------------------------------------------------------
    # Dedicated FinCo Loggers
    # --------------------------------------------------------

    logging.getLogger(
        "finco.audit"
    ).setLevel(logging.INFO)

    logging.getLogger(
        "finco.security"
    ).setLevel(logging.INFO)

    logging.getLogger(
        "finco.agent"
    ).setLevel(logging.INFO)

    logging.getLogger(
        "finco.rag"
    ).setLevel(logging.INFO)

    logging.getLogger(
        "finco.ml"
    ).setLevel(logging.INFO)

    _configured = True


# ============================================================
# Convenience Logging Functions
# ============================================================

def log_event(
    logger: logging.Logger,
    message: str,
    level: int = logging.INFO,
    **metadata: Any,
) -> None:
    """
    Log a structured event.

    Example:
        log_event(
            logger,
            "Forecast generated",
            forecast_id="123",
            horizon=12,
        )
    """

    sanitized = _sanitize_value(
        metadata
    )

    logger.log(
        level,
        message,
        extra=sanitized,
    )


def log_exception(
    logger: logging.Logger,
    message: str,
    **metadata: Any,
) -> None:
    """
    Log an exception with structured metadata.
    """

    sanitized = _sanitize_value(
        metadata
    )

    logger.exception(
        message,
        extra=sanitized,
    )


# ============================================================
# Startup Logging
# ============================================================

def log_startup() -> None:
    """Log FinCo AI application startup."""

    logger = get_logger("finco.startup")

    logger.info(
        "FinCo AI application started",
        extra={
            "environment": ENVIRONMENT,
            "log_level": LOG_LEVEL,
        },
    )


def log_shutdown() -> None:
    """Log FinCo AI application shutdown."""

    logger = get_logger("finco.shutdown")

    logger.info(
        "FinCo AI application shutting down"
    )


# ============================================================
# Automatic Configuration
# ============================================================

configure_logging()