"""
backend/app/api/transactions.py

Transaction API for FinCo AI.

Responsibilities
----------------
- Authentication / authorization
- Tenant isolation
- Transaction request validation
- Transaction creation
- Transaction retrieval
- Transaction filtering
- Transaction summaries
- Transaction statistics
- Fraud-analysis handoff
- Safe error handling

Architecture
------------

    Client
       |
       v
    FastAPI Transaction API
       |
       +----------------------+
       |                      |
       v                      v
TransactionService      TransactionRepository
       |                      |
       v                      v
Financial Engine          PostgreSQL
       |
       +---- Fraud Detection
       +---- Revenue
       +---- Expenses
       +---- Cash Flow
       +---- Alerts
       +---- Risk
       |
       v
    FinCo AI Intelligence
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any, Literal

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    status,
)
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/transactions",
    tags=["Transactions"],
)


# ============================================================================
# TYPES
# ============================================================================

TransactionType = Literal[
    "income",
    "expense",
    "transfer",
    "refund",
    "adjustment",
]

TransactionStatus = Literal[
    "pending",
    "completed",
    "failed",
    "cancelled",
    "reversed",
]

RiskLevel = Literal[
    "very_low",
    "low",
    "medium",
    "high",
    "critical",
]


# ============================================================================
# REQUEST SCHEMAS
# ============================================================================


class TransactionCreateRequest(BaseModel):
    """
    Create a financial transaction.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    transaction_date: date

    transaction_type: TransactionType

    amount: Decimal = Field(
        ...,
        gt=0,
    )

    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=10,
    )

    category: str | None = Field(
        default=None,
        max_length=128,
    )

    subcategory: str | None = Field(
        default=None,
        max_length=128,
    )

    description: str | None = Field(
        default=None,
        max_length=1000,
    )

    account_id: str | None = Field(
        default=None,
        max_length=128,
    )

    customer_id: str | None = Field(
        default=None,
        max_length=128,
    )

    vendor_id: str | None = Field(
        default=None,
        max_length=128,
    )

    reference: str | None = Field(
        default=None,
        max_length=256,
    )

    external_transaction_id: str | None = Field(
        default=None,
        max_length=256,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: str,
    ) -> str:
        return value.upper().strip()


class TransactionUpdateRequest(BaseModel):
    """
    Update mutable transaction attributes.
    """

    model_config = ConfigDict(extra="forbid")

    category: str | None = Field(
        default=None,
        max_length=128,
    )

    subcategory: str | None = Field(
        default=None,
        max_length=128,
    )

    description: str | None = Field(
        default=None,
        max_length=1000,
    )

    status: TransactionStatus | None = None

    metadata: dict[str, Any] | None = None


class TransactionBatchRequest(BaseModel):
    """
    Batch transaction ingestion.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    transactions: list[TransactionCreateRequest] = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    run_fraud_detection: bool = True

    run_normalization: bool = True


class TransactionFilterRequest(BaseModel):
    """
    Structured transaction filtering.
    """

    model_config = ConfigDict(extra="forbid")

    company_id: str

    start_date: date | None = None

    end_date: date | None = None

    transaction_type: TransactionType | None = None

    transaction_status: TransactionStatus | None = None

    category: str | None = None

    currency: str | None = None

    min_amount: Decimal | None = Field(
        default=None,
        ge=0,
    )

    max_amount: Decimal | None = Field(
        default=None,
        ge=0,
    )

    search: str | None = Field(
        default=None,
        max_length=256,
    )


# ============================================================================
# RESPONSE SCHEMAS
# ============================================================================


class TransactionResponse(BaseModel):
    """
    Transaction representation returned by the API.
    """

    model_config = ConfigDict(
        extra="ignore",
        from_attributes=True,
    )

    id: str

    company_id: str

    transaction_date: date

    transaction_type: TransactionType

    amount: Decimal

    currency: str

    category: str | None = None

    subcategory: str | None = None

    description: str | None = None

    account_id: str | None = None

    customer_id: str | None = None

    vendor_id: str | None = None

    reference: str | None = None

    external_transaction_id: str | None = None

    status: TransactionStatus

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    fraud_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    risk_level: RiskLevel | None = None

    created_at: datetime

    updated_at: datetime


class TransactionListResponse(BaseModel):
    """
    Paginated transaction response.
    """

    company_id: str

    items: list[TransactionResponse]

    total: int

    limit: int

    offset: int

    has_more: bool


class TransactionBatchResponse(BaseModel):
    """
    Result of batch transaction ingestion.
    """

    company_id: str

    received: int

    created: int

    failed: int

    transaction_ids: list[str] = Field(
        default_factory=list,
    )

    errors: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    fraud_detection_requested: bool

    normalization_requested: bool

    processed_at: datetime


class TransactionSummaryResponse(BaseModel):
    """
    Transaction aggregation.
    """

    company_id: str

    period_start: date

    period_end: date

    currency: str | None = None

    total_transactions: int

    total_income: Decimal

    total_expenses: Decimal

    total_transfers: Decimal

    total_refunds: Decimal

    net_cash_movement: Decimal

    average_transaction: Decimal

    largest_transaction: Decimal

    generated_at: datetime


class TransactionCategorySummary(BaseModel):
    """
    Aggregated transaction category.
    """

    category: str

    transaction_count: int

    total_amount: Decimal

    percentage_of_total: float


class TransactionCategoryResponse(BaseModel):
    """
    Transaction category analysis.
    """

    company_id: str

    period_start: date

    period_end: date

    categories: list[TransactionCategorySummary]

    generated_at: datetime


class TransactionStatisticsResponse(BaseModel):
    """
    Transaction statistics used by financial and fraud engines.
    """

    company_id: str

    period_start: date

    period_end: date

    total_transactions: int

    completed_transactions: int

    pending_transactions: int

    failed_transactions: int

    cancelled_transactions: int

    total_amount: Decimal

    average_amount: Decimal

    median_amount: Decimal | None = None

    largest_amount: Decimal

    smallest_amount: Decimal

    income_count: int

    expense_count: int

    transfer_count: int

    refund_count: int

    high_risk_transactions: int = 0

    critical_risk_transactions: int = 0

    generated_at: datetime


class TransactionFraudRequest(BaseModel):
    """
    Request to analyze an existing transaction for fraud.
    """

    model_config = ConfigDict(extra="forbid")

    transaction_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    persist_result: bool = True


class TransactionFraudResponse(BaseModel):
    """
    Fraud-analysis result for a transaction.
    """

    transaction_id: str

    company_id: str

    fraud_probability: float = Field(
        ge=0,
        le=1,
    )

    risk_level: RiskLevel

    decision: Literal[
        "allow",
        "review",
        "block",
        "unknown",
    ]

    factors: list[dict[str, Any]] = Field(
        default_factory=list,
    )

    explanation: str | None = None

    analyzed_at: datetime


class TransactionHealthResponse(BaseModel):
    """
    Transaction service health.
    """

    status: Literal[
        "healthy",
        "degraded",
        "unhealthy",
    ]

    service: str

    timestamp: datetime

    details: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# AUTHENTICATION
# ============================================================================


def get_current_user() -> Any:
    """
    Lazily load authentication dependency.
    """

    try:
        from backend.app.dependencies import (
            get_current_user as dependency,
        )

        return dependency

    except ImportError as exc:
        logger.exception(
            "Authentication dependency unavailable."
        )

        raise RuntimeError(
            "Authentication dependency is not configured."
        ) from exc


# ============================================================================
# SERVICES
# ============================================================================


def get_transaction_service() -> Any:
    """
    Lazily construct TransactionService.

    Expected implementation:

        backend.app.transactions.transaction_service.TransactionService

    A repository fallback is supported for incremental development.
    """

    try:
        from backend.app.transactions.transaction_service import (
            TransactionService,
        )

        return TransactionService()

    except ImportError:
        pass

    try:
        from backend.app.database.repositories.transaction_repository import (
            TransactionRepository,
        )

        return TransactionRepository()

    except ImportError as exc:
        logger.exception(
            "Transaction service/repository unavailable."
        )

        raise RuntimeError(
            "Transaction service is not configured."
        ) from exc


def get_fraud_service() -> Any:
    """
    Lazily load the fraud service.
    """

    try:
        from backend.app.fraud.fraud_service import (
            FraudService,
        )

        return FraudService()

    except ImportError as exc:
        logger.exception(
            "Fraud service unavailable."
        )

        raise RuntimeError(
            "Fraud service is not configured."
        ) from exc


# ============================================================================
# SECURITY
# ============================================================================


def _get_user_company_id(
    current_user: Any,
) -> str | None:
    """
    Extract company ID from authenticated user.
    """

    if current_user is None:
        return None

    if isinstance(current_user, dict):
        return current_user.get("company_id")

    return getattr(
        current_user,
        "company_id",
        None,
    )


def _is_admin(
    current_user: Any,
) -> bool:
    """
    Determine administrative access.
    """

    if current_user is None:
        return False

    if isinstance(current_user, dict):
        return bool(
            current_user.get(
                "is_admin",
                False,
            )
        )

    return bool(
        getattr(
            current_user,
            "is_admin",
            False,
        )
    )


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce tenant isolation.
    """

    if _is_admin(current_user):
        return

    user_company_id = _get_user_company_id(
        current_user
    )

    if not user_company_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "User is not associated with a company."
            ),
        )

    if user_company_id != company_id:
        logger.warning(
            "Unauthorized transaction access attempt: "
            "user_company=%s requested_company=%s",
            user_company_id,
            company_id,
        )

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                "You do not have access to this "
                "company's transactions."
            ),
        )


# ============================================================================
# VALIDATION
# ============================================================================


def validate_period(
    start_date: date | None,
    end_date: date | None,
    max_days: int = 3660,
) -> None:
    """
    Validate transaction date range.
    """

    if start_date is None or end_date is None:
        return

    if start_date > end_date:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "start_date cannot be after end_date."
            ),
        )

    if (end_date - start_date).days > max_days:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"Requested period cannot exceed "
                f"{max_days} days."
            ),
        )


def validate_amount_range(
    min_amount: Decimal | None,
    max_amount: Decimal | None,
) -> None:
    """
    Validate amount filtering.
    """

    if (
        min_amount is not None
        and max_amount is not None
        and min_amount > max_amount
    ):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "min_amount cannot be greater than "
                "max_amount."
            ),
        )


# ============================================================================
# SERVICE CALLER
# ============================================================================


async def _call_service(
    service: Any,
    method_name: str,
    **kwargs: Any,
) -> Any:
    """
    Invoke service method.

    Supports synchronous and asynchronous implementations.
    """

    method = getattr(
        service,
        method_name,
        None,
    )

    if method is None:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=(
                f"Transaction service does not implement "
                f"'{method_name}'."
            ),
        )

    try:
        result = method(**kwargs)

        if hasattr(result, "__await__"):
            result = await result

        return result

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid transaction request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid transaction request.",
        ) from exc

    except Exception as exc:
        logger.exception(
            "Transaction service operation failed: %s",
            method_name,
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transaction operation failed.",
        ) from exc


# ============================================================================
# CREATE TRANSACTION
# ============================================================================


@router.post(
    "",
    response_model=TransactionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create transaction",
)
async def create_transaction(
    request: TransactionCreateRequest,
    current_user: Any = Depends(get_current_user()),
) -> TransactionResponse:
    """
    Create a financial transaction.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "create",
        company_id=request.company_id,
        transaction_date=request.transaction_date,
        transaction_type=request.transaction_type,
        amount=request.amount,
        currency=request.currency,
        category=request.category,
        subcategory=request.subcategory,
        description=request.description,
        account_id=request.account_id,
        customer_id=request.customer_id,
        vendor_id=request.vendor_id,
        reference=request.reference,
        external_transaction_id=(
            request.external_transaction_id
        ),
        metadata=request.metadata,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Transaction could not be created.",
        )

    return TransactionResponse.model_validate(
        result
    )


# ============================================================================
# BATCH CREATE
# ============================================================================


@router.post(
    "/batch",
    response_model=TransactionBatchResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create transactions in batch",
)
async def create_transactions_batch(
    request: TransactionBatchRequest,
    current_user: Any = Depends(get_current_user()),
) -> TransactionBatchResponse:
    """
    Ingest multiple transactions.

    Suitable for:
        - CSV imports
        - Excel imports
        - bank feeds
        - ERP synchronization
        - accounting-system ingestion
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    for transaction in request.transactions:
        if transaction.company_id != request.company_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "All transactions in a batch must "
                    "belong to the requested company."
                ),
            )

    service = get_transaction_service()

    payload = [
        transaction.model_dump(
            mode="json"
        )
        for transaction in request.transactions
    ]

    result = await _call_service(
        service,
        "create_batch",
        company_id=request.company_id,
        transactions=payload,
        run_fraud_detection=(
            request.run_fraud_detection
        ),
        run_normalization=(
            request.run_normalization
        ),
        user=current_user,
    )

    if isinstance(
        result,
        TransactionBatchResponse,
    ):
        return result

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch transaction ingestion failed.",
        )

    return TransactionBatchResponse.model_validate(
        result
    )


# ============================================================================
# GET TRANSACTION
# ============================================================================


@router.get(
    "/{transaction_id}",
    response_model=TransactionResponse,
    summary="Get transaction",
)
async def get_transaction(
    transaction_id: str,
    company_id: str = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> TransactionResponse:
    """
    Retrieve one transaction.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "get",
        transaction_id=transaction_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found.",
        )

    return TransactionResponse.model_validate(
        result
    )


# ============================================================================
# LIST TRANSACTIONS
# ============================================================================


@router.get(
    "",
    response_model=TransactionListResponse,
    summary="List transactions",
)
async def list_transactions(
    company_id: str = Query(...),
    start_date: date | None = Query(
        default=None
    ),
    end_date: date | None = Query(
        default=None
    ),
    transaction_type: TransactionType | None = Query(
        default=None
    ),
    transaction_status: TransactionStatus | None = Query(
        default=None
    ),
    category: str | None = Query(
        default=None,
        max_length=128,
    ),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    min_amount: Decimal | None = Query(
        default=None,
        ge=0,
    ),
    max_amount: Decimal | None = Query(
        default=None,
        ge=0,
    ),
    search: str | None = Query(
        default=None,
        max_length=256,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    current_user: Any = Depends(get_current_user()),
) -> TransactionListResponse:
    """
    List transactions with secure filtering and pagination.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    validate_amount_range(
        min_amount,
        max_amount,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "list",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        transaction_type=transaction_type,
        transaction_status=transaction_status,
        category=category,
        currency=currency.upper()
        if currency
        else None,
        min_amount=min_amount,
        max_amount=max_amount,
        search=search,
        limit=limit,
        offset=offset,
        user=current_user,
    )

    if isinstance(
        result,
        TransactionListResponse,
    ):
        return result

    if result is None:
        return TransactionListResponse(
            company_id=company_id,
            items=[],
            total=0,
            limit=limit,
            offset=offset,
            has_more=False,
        )

    return TransactionListResponse.model_validate(
        result
    )


# ============================================================================
# UPDATE TRANSACTION
# ============================================================================


@router.patch(
    "/{transaction_id}",
    response_model=TransactionResponse,
    summary="Update transaction",
)
async def update_transaction(
    transaction_id: str,
    request: TransactionUpdateRequest,
    company_id: str = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> TransactionResponse:
    """
    Update mutable transaction fields.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "update",
        transaction_id=transaction_id,
        company_id=company_id,
        updates=request.model_dump(
            exclude_unset=True
        ),
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found.",
        )

    return TransactionResponse.model_validate(
        result
    )


# ============================================================================
# DELETE / VOID TRANSACTION
# ============================================================================


@router.delete(
    "/{transaction_id}",
    summary="Void transaction",
)
async def void_transaction(
    transaction_id: str,
    company_id: str = Query(...),
    current_user: Any = Depends(get_current_user()),
) -> dict[str, Any]:
    """
    Void a transaction.

    Financial systems should normally use a reversal/void
    operation rather than physically deleting accounting data.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "void",
        transaction_id=transaction_id,
        company_id=company_id,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found.",
        )

    return {
        "transaction_id": transaction_id,
        "company_id": company_id,
        "status": "voided",
        "data": result,
        "timestamp": datetime.now(
            timezone.utc
        ),
    }


# ============================================================================
# SUMMARY
# ============================================================================


@router.get(
    "/summary/{company_id}",
    response_model=TransactionSummaryResponse,
    summary="Get transaction summary",
)
async def get_transaction_summary(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> TransactionSummaryResponse:
    """
    Aggregate transactions for a reporting period.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "summary",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency.upper()
        if currency
        else None,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction summary not found.",
        )

    if isinstance(
        result,
        TransactionSummaryResponse,
    ):
        return result

    return TransactionSummaryResponse.model_validate(
        result
    )


# ============================================================================
# CATEGORY ANALYSIS
# ============================================================================


@router.get(
    "/categories/{company_id}",
    response_model=TransactionCategoryResponse,
    summary="Get transaction categories",
)
async def get_transaction_categories(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    transaction_type: TransactionType | None = Query(
        default=None
    ),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> TransactionCategoryResponse:
    """
    Return transaction amounts grouped by category.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "categories",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        transaction_type=transaction_type,
        currency=currency.upper()
        if currency
        else None,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction category data not found.",
        )

    if isinstance(
        result,
        TransactionCategoryResponse,
    ):
        return result

    return TransactionCategoryResponse.model_validate(
        result
    )


# ============================================================================
# STATISTICS
# ============================================================================


@router.get(
    "/statistics/{company_id}",
    response_model=TransactionStatisticsResponse,
    summary="Get transaction statistics",
)
async def get_transaction_statistics(
    company_id: str,
    start_date: date = Query(...),
    end_date: date = Query(...),
    currency: str | None = Query(
        default=None,
        min_length=3,
        max_length=10,
    ),
    current_user: Any = Depends(get_current_user()),
) -> TransactionStatisticsResponse:
    """
    Return transaction statistics.

    Used by:
        - financial analysis
        - fraud detection
        - anomaly detection
        - risk scoring
        - forecasting
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "statistics",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        currency=currency.upper()
        if currency
        else None,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction statistics not found.",
        )

    if isinstance(
        result,
        TransactionStatisticsResponse,
    ):
        return result

    return TransactionStatisticsResponse.model_validate(
        result
    )


# ============================================================================
# FRAUD ANALYSIS
# ============================================================================


@router.post(
    "/fraud",
    response_model=TransactionFraudResponse,
    summary="Analyze transaction for fraud",
)
async def analyze_transaction_fraud(
    request: TransactionFraudRequest,
    current_user: Any = Depends(get_current_user()),
) -> TransactionFraudResponse:
    """
    Send an existing transaction to the fraud engine.

    Important:
        A fraud probability is a risk signal, not proof of
        fraudulent activity. High-risk results should normally
        enter human-review workflows.
    """

    validate_company_access(
        request.company_id,
        current_user,
    )

    transaction_service = get_transaction_service()

    transaction = await _call_service(
        transaction_service,
        "get",
        transaction_id=request.transaction_id,
        company_id=request.company_id,
        user=current_user,
    )

    if transaction is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Transaction not found.",
        )

    fraud_service = get_fraud_service()

    result = await _call_service(
        fraud_service,
        "score_transaction",
        transaction=transaction,
        company_id=request.company_id,
        persist_result=request.persist_result,
        user=current_user,
    )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Fraud analysis failed.",
        )

    if isinstance(
        result,
        TransactionFraudResponse,
    ):
        return result

    return TransactionFraudResponse.model_validate(
        result
    )


# ============================================================================
# HIGH-RISK TRANSACTIONS
# ============================================================================


@router.get(
    "/high-risk/{company_id}",
    response_model=TransactionListResponse,
    summary="Get high-risk transactions",
)
async def get_high_risk_transactions(
    company_id: str,
    start_date: date | None = Query(
        default=None
    ),
    end_date: date | None = Query(
        default=None
    ),
    minimum_fraud_score: float = Query(
        default=0.80,
        ge=0.0,
        le=1.0,
    ),
    limit: int = Query(
        default=100,
        ge=1,
        le=1000,
    ),
    offset: int = Query(
        default=0,
        ge=0,
    ),
    current_user: Any = Depends(get_current_user()),
) -> TransactionListResponse:
    """
    Retrieve transactions with elevated fraud/risk scores.
    """

    validate_company_access(
        company_id,
        current_user,
    )

    validate_period(
        start_date,
        end_date,
    )

    service = get_transaction_service()

    result = await _call_service(
        service,
        "high_risk",
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        minimum_fraud_score=minimum_fraud_score,
        limit=limit,
        offset=offset,
        user=current_user,
    )

    if isinstance(
        result,
        TransactionListResponse,
    ):
        return result

    if result is None:
        return TransactionListResponse(
            company_id=company_id,
            items=[],
            total=0,
            limit=limit,
            offset=offset,
            has_more=False,
        )

    return TransactionListResponse.model_validate(
        result
    )


# ============================================================================
# HEALTH
# ============================================================================


@router.get(
    "/health",
    response_model=TransactionHealthResponse,
    summary="Transaction service health",
)
async def transaction_health() -> TransactionHealthResponse:
    """
    Health check for the transaction subsystem.
    """

    started = time.perf_counter()

    try:
        service = get_transaction_service()

        health_method = getattr(
            service,
            "health",
            None,
        )

        details: dict[str, Any] = {}

        if health_method is not None:
            result = health_method()

            if hasattr(result, "__await__"):
                result = await result

            if isinstance(result, dict):
                details = result

        details["latency_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )

        return TransactionHealthResponse(
            status="healthy",
            service="transactions",
            timestamp=datetime.now(
                timezone.utc
            ),
            details=details,
        )

    except Exception:
        logger.exception(
            "Transaction service health check failed."
        )

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Transaction service is unavailable.",
        )