# backend/app/core/exceptions.py

from __future__ import annotations

from typing import Any, Optional


class FinCoException(Exception):
    """
    Base exception for all application-specific FinCo AI errors.
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "FINCO_ERROR",
        status_code: int = 500,
        details: Optional[dict[str, Any]] = None,
    ):
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}

        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert exception into a JSON-friendly response.
        """

        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
        }


# ======================================================================
# 400 - BAD REQUEST
# ======================================================================


class BadRequestError(FinCoException):
    """Raised when the client sends invalid input."""

    def __init__(
        self,
        message: str = "Invalid request.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="BAD_REQUEST",
            status_code=400,
            details=details,
        )


class ValidationError(FinCoException):
    """Raised when business-level validation fails."""

    def __init__(
        self,
        message: str = "Validation failed.",
        *,
        field: Optional[str] = None,
        details: Optional[dict[str, Any]] = None,
    ):
        validation_details = details or {}

        if field:
            validation_details["field"] = field

        super().__init__(
            message,
            code="VALIDATION_ERROR",
            status_code=400,
            details=validation_details,
        )


# ======================================================================
# 401 - AUTHENTICATION
# ======================================================================


class AuthenticationError(FinCoException):
    """Raised when authentication fails."""

    def __init__(
        self,
        message: str = "Authentication required.",
    ):
        super().__init__(
            message,
            code="AUTHENTICATION_ERROR",
            status_code=401,
        )


class InvalidTokenError(AuthenticationError):
    """Raised when a JWT/access token is invalid."""

    def __init__(
        self,
        message: str = "Invalid or expired authentication token.",
    ):
        super().__init__(message)
        self.code = "INVALID_TOKEN"


class CredentialsError(AuthenticationError):
    """Raised when credentials are invalid."""

    def __init__(
        self,
        message: str = "Invalid credentials.",
    ):
        super().__init__(message)
        self.code = "INVALID_CREDENTIALS"


# ======================================================================
# 403 - AUTHORIZATION
# ======================================================================


class AuthorizationError(FinCoException):
    """Raised when the authenticated user lacks permission."""

    def __init__(
        self,
        message: str = "You do not have permission to perform this action.",
        *,
        permission: Optional[str] = None,
    ):
        details = {}

        if permission:
            details["permission"] = permission

        super().__init__(
            message,
            code="AUTHORIZATION_ERROR",
            status_code=403,
            details=details,
        )


class PermissionDeniedError(AuthorizationError):
    """Raised when RBAC permission checks fail."""

    def __init__(
        self,
        message: str = "Permission denied.",
        *,
        permission: Optional[str] = None,
    ):
        super().__init__(
            message,
            permission=permission,
        )
        self.code = "PERMISSION_DENIED"


class TenantAccessError(AuthorizationError):
    """
    Raised when a user attempts to access another company's data.

    Important for FinCo AI multi-tenant isolation.
    """

    def __init__(
        self,
        message: str = "Access to this company's data is not allowed.",
    ):
        super().__init__(message)
        self.code = "TENANT_ACCESS_DENIED"


# ======================================================================
# 404 - NOT FOUND
# ======================================================================


class NotFoundError(FinCoException):
    """Base error for missing resources."""

    def __init__(
        self,
        resource: str,
        identifier: Optional[Any] = None,
    ):
        message = f"{resource} not found."

        if identifier is not None:
            message = f"{resource} with id '{identifier}' not found."

        super().__init__(
            message,
            code="NOT_FOUND",
            status_code=404,
            details={
                "resource": resource,
                "identifier": identifier,
            },
        )


class UserNotFoundError(NotFoundError):
    def __init__(self, user_id: Optional[int] = None):
        super().__init__("User", user_id)


class CompanyNotFoundError(NotFoundError):
    def __init__(self, company_id: Optional[int] = None):
        super().__init__("Company", company_id)


class DocumentNotFoundError(NotFoundError):
    def __init__(self, document_id: Optional[int] = None):
        super().__init__("Document", document_id)


class TransactionNotFoundError(NotFoundError):
    def __init__(self, transaction_id: Optional[int] = None):
        super().__init__("Transaction", transaction_id)


class AlertNotFoundError(NotFoundError):
    def __init__(self, alert_id: Optional[int] = None):
        super().__init__("Alert", alert_id)


class RecommendationNotFoundError(NotFoundError):
    def __init__(self, recommendation_id: Optional[int] = None):
        super().__init__("Recommendation", recommendation_id)


# ======================================================================
# 409 - CONFLICT
# ======================================================================


class ConflictError(FinCoException):
    """Raised when an operation conflicts with existing data."""

    def __init__(
        self,
        message: str = "Resource conflict.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="CONFLICT",
            status_code=409,
            details=details,
        )


class DuplicateResourceError(ConflictError):
    """Raised when attempting to create a duplicate resource."""

    def __init__(
        self,
        resource: str,
        field: Optional[str] = None,
    ):
        details = {}

        if field:
            details["field"] = field

        super().__init__(
            f"{resource} already exists.",
            details=details,
        )
        self.code = "DUPLICATE_RESOURCE"


class DuplicateTransactionError(ConflictError):
    """Raised when a transaction reference already exists."""

    def __init__(
        self,
        reference_number: str,
    ):
        super().__init__(
            "Transaction with this reference already exists.",
            details={
                "reference_number": reference_number,
            },
        )
        self.code = "DUPLICATE_TRANSACTION"


# ======================================================================
# 413 - FILE TOO LARGE
# ======================================================================


class FileTooLargeError(FinCoException):
    """Raised when an uploaded file exceeds the configured limit."""

    def __init__(
        self,
        max_size_mb: Optional[int] = None,
    ):
        details = {}

        if max_size_mb is not None:
            details["max_size_mb"] = max_size_mb

        super().__init__(
            "Uploaded file exceeds the allowed size.",
            code="FILE_TOO_LARGE",
            status_code=413,
            details=details,
        )


# ======================================================================
# 415 - UNSUPPORTED FILE
# ======================================================================


class UnsupportedFileTypeError(FinCoException):
    """Raised when an uploaded file type is unsupported."""

    def __init__(
        self,
        file_type: Optional[str] = None,
    ):
        details = {}

        if file_type:
            details["file_type"] = file_type

        super().__init__(
            "Unsupported file type.",
            code="UNSUPPORTED_FILE_TYPE",
            status_code=415,
            details=details,
        )


# ======================================================================
# 422 - FINANCIAL DATA
# ======================================================================


class FinancialDataError(FinCoException):
    """Base exception for financial-data processing errors."""

    def __init__(
        self,
        message: str = "Financial data processing failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="FINANCIAL_DATA_ERROR",
            status_code=422,
            details=details,
        )


class InvalidFinancialDataError(FinancialDataError):
    """Raised when financial data contains invalid values."""

    def __init__(
        self,
        message: str = "Invalid financial data.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            details=details,
        )
        self.code = "INVALID_FINANCIAL_DATA"


class MissingFinancialDataError(FinancialDataError):
    """Raised when required financial data is unavailable."""

    def __init__(
        self,
        fields: Optional[list[str]] = None,
    ):
        super().__init__(
            "Required financial data is missing.",
            details={
                "missing_fields": fields or [],
            },
        )
        self.code = "MISSING_FINANCIAL_DATA"


# ======================================================================
# RAG
# ======================================================================


class RAGError(FinCoException):
    """Base exception for Retrieval-Augmented Generation."""

    def __init__(
        self,
        message: str = "RAG operation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="RAG_ERROR",
            status_code=500,
            details=details,
        )


class DocumentProcessingError(RAGError):
    """Raised when document parsing/chunking fails."""

    def __init__(
        self,
        message: str = "Document processing failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            details=details,
        )
        self.code = "DOCUMENT_PROCESSING_ERROR"


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""

    def __init__(
        self,
        message: str = "Embedding generation failed.",
    ):
        super().__init__(message)
        self.code = "EMBEDDING_ERROR"


class RetrievalError(RAGError):
    """Raised when retrieval fails."""

    def __init__(
        self,
        message: str = "Document retrieval failed.",
    ):
        super().__init__(message)
        self.code = "RETRIEVAL_ERROR"


class CitationError(RAGError):
    """Raised when generated citations are invalid."""

    def __init__(
        self,
        message: str = "Citation validation failed.",
    ):
        super().__init__(message)
        self.code = "CITATION_ERROR"


# ======================================================================
# ML / FORECASTING
# ======================================================================


class MLError(FinCoException):
    """Base exception for ML operations."""

    def __init__(
        self,
        message: str = "Machine learning operation failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="ML_ERROR",
            status_code=500,
            details=details,
        )


class ModelNotFoundError(MLError):
    """Raised when a trained model cannot be located."""

    def __init__(
        self,
        model_name: str,
    ):
        super().__init__(
            f"ML model '{model_name}' was not found.",
            details={
                "model_name": model_name,
            },
        )
        self.code = "MODEL_NOT_FOUND"


class ModelPredictionError(MLError):
    """Raised when model inference fails."""

    def __init__(
        self,
        model_name: Optional[str] = None,
    ):
        details = {}

        if model_name:
            details["model_name"] = model_name

        super().__init__(
            "Model prediction failed.",
            details=details,
        )
        self.code = "MODEL_PREDICTION_ERROR"


class ForecastError(MLError):
    """Raised when financial forecasting fails."""

    def __init__(
        self,
        message: str = "Financial forecast failed.",
    ):
        super().__init__(message)
        self.code = "FORECAST_ERROR"


# ======================================================================
# FRAUD
# ======================================================================


class FraudDetectionError(MLError):
    """Raised when fraud detection cannot be completed."""

    def __init__(
        self,
        message: str = "Fraud detection failed.",
    ):
        super().__init__(message)
        self.code = "FRAUD_DETECTION_ERROR"


# ======================================================================
# AGENTIC AI
# ======================================================================


class AgentError(FinCoException):
    """Base exception for Agentic AI failures."""

    def __init__(
        self,
        message: str = "Agent execution failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="AGENT_ERROR",
            status_code=500,
            details=details,
        )


class AgentRoutingError(AgentError):
    """Raised when the supervisor cannot route a task."""

    def __init__(
        self,
        message: str = "Unable to route request to an agent.",
    ):
        super().__init__(message)
        self.code = "AGENT_ROUTING_ERROR"


class AgentPlanningError(AgentError):
    """Raised when an agent cannot create a valid plan."""

    def __init__(
        self,
        message: str = "Agent planning failed.",
    ):
        super().__init__(message)
        self.code = "AGENT_PLANNING_ERROR"


class ToolExecutionError(AgentError):
    """Raised when an agent tool fails."""

    def __init__(
        self,
        tool_name: str,
        message: str = "Agent tool execution failed.",
    ):
        super().__init__(
            message,
            details={
                "tool_name": tool_name,
            },
        )
        self.code = "TOOL_EXECUTION_ERROR"


class AgentGuardrailError(AgentError):
    """Raised when an agent violates a configured guardrail."""

    def __init__(
        self,
        message: str = "Agent request blocked by safety guardrails.",
    ):
        super().__init__(message)
        self.code = "AGENT_GUARDRAIL_BLOCKED"
        self.status_code = 400


# ======================================================================
# ALERTS / RISK
# ======================================================================


class AlertError(FinCoException):
    """Base exception for alert-engine failures."""

    def __init__(
        self,
        message: str = "Alert processing failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="ALERT_ERROR",
            status_code=500,
            details=details,
        )


class RiskScoringError(AlertError):
    """Raised when financial risk scoring fails."""

    def __init__(
        self,
        message: str = "Risk scoring failed.",
    ):
        super().__init__(message)
        self.code = "RISK_SCORING_ERROR"


class AlertGenerationError(AlertError):
    """Raised when an alert cannot be generated."""

    def __init__(
        self,
        message: str = "Alert generation failed.",
    ):
        super().__init__(message)
        self.code = "ALERT_GENERATION_ERROR"


# ======================================================================
# WHAT-IF
# ======================================================================


class WhatIfError(FinCoException):
    """Base exception for scenario simulation."""

    def __init__(
        self,
        message: str = "What-if analysis failed.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            code="WHAT_IF_ERROR",
            status_code=422,
            details=details,
        )


class InvalidScenarioError(WhatIfError):
    """Raised when scenario parameters are invalid."""

    def __init__(
        self,
        message: str = "Invalid financial scenario.",
        *,
        details: Optional[dict[str, Any]] = None,
    ):
        super().__init__(
            message,
            details=details,
        )
        self.code = "INVALID_SCENARIO"


# ======================================================================
# RATE LIMITING
# ======================================================================


class RateLimitExceededError(FinCoException):
    """Raised when a user/client exceeds API limits."""

    def __init__(
        self,
        retry_after: Optional[int] = None,
    ):
        details = {}

        if retry_after is not None:
            details["retry_after_seconds"] = retry_after

        super().__init__(
            "Rate limit exceeded. Please try again later.",
            code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details=details,
        )


# ======================================================================
# DATABASE
# ======================================================================


class DatabaseError(FinCoException):
    """Base database exception."""

    def __init__(
        self,
        message: str = "Database operation failed.",
    ):
        super().__init__(
            message,
            code="DATABASE_ERROR",
            status_code=500,
        )


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    def __init__(
        self,
        message: str = "Unable to connect to the database.",
    ):
        super().__init__(message)
        self.code = "DATABASE_CONNECTION_ERROR"


# ======================================================================
# EXTERNAL SERVICES
# ======================================================================


class ExternalServiceError(FinCoException):
    """Base exception for external service failures."""

    def __init__(
        self,
        service: str,
        message: str = "External service unavailable.",
    ):
        super().__init__(
            message,
            code="EXTERNAL_SERVICE_ERROR",
            status_code=502,
            details={
                "service": service,
            },
        )


class LLMServiceError(ExternalServiceError):
    """Raised when the LLM provider fails."""

    def __init__(
        self,
        message: str = "LLM service unavailable.",
    ):
        super().__init__(
            service="llm",
            message=message,
        )
        self.code = "LLM_SERVICE_ERROR"


class VectorDatabaseError(ExternalServiceError):
    """Raised when vector database operations fail."""

    def __init__(
        self,
        message: str = "Vector database operation failed.",
    ):
        super().__init__(
            service="vector_database",
            message=message,
        )
        self.code = "VECTOR_DATABASE_ERROR"


# ======================================================================
# HUMAN REVIEW
# ======================================================================


class HumanReviewRequiredError(FinCoException):
    """
    Indicates that an AI-generated decision requires human approval.

    This should generally be treated as a workflow state rather than
    an unexpected system failure.
    """

    def __init__(
        self,
        message: str = "Human review is required before proceeding.",
        *,
        entity_type: Optional[str] = None,
        entity_id: Optional[int] = None,
    ):
        details = {
            "requires_human_review": True,
        }

        if entity_type:
            details["entity_type"] = entity_type

        if entity_id is not None:
            details["entity_id"] = entity_id

        super().__init__(
            message,
            code="HUMAN_REVIEW_REQUIRED",
            status_code=409,
            details=details,
        )