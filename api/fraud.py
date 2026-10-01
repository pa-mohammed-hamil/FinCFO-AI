"""
FinCo AI - Fraud Detection API

API endpoints for transaction fraud detection and investigation.

Architecture:

    Client
      ↓
    fraud.py
      ↓
    FraudService
      ↓
    ┌──────────────────────────┐
    │ Preprocessing            │
    │ Feature Engineering      │
    │ Supervised Model         │
    │ Anomaly Detection        │
    │ Risk Scoring             │
    │ Explainability           │
    └──────────────────────────┘
      ↓
    Fraud Alert
      ↓
    Alert Engine / Supervisor Agent
      ↓
    Human Review / Audit Log

The API layer must not contain model-training logic.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field

from backend.app.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/fraud",
    tags=["Fraud Detection"],
)


# ============================================================
# Types
# ============================================================

TransactionType = Literal[
    "payment",
    "purchase",
    "refund",
    "transfer",
    "withdrawal",
    "deposit",
    "invoice",
    "other",
]

RiskLevel = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
    "CRITICAL",
]

FraudDecision = Literal[
    "LEGITIMATE",
    "SUSPICIOUS",
    "FRAUD",
    "REVIEW",
]


# ============================================================
# Request Schemas
# ============================================================


class FraudTransactionRequest(BaseModel):
    """
    Request for single transaction fraud analysis.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    transaction_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    customer_id: str | None = Field(
        default=None,
        max_length=128,
    )

    amount: Decimal = Field(
        ...,
        ge=0,
    )

    currency: str = Field(
        "USD",
        min_length=3,
        max_length=3,
    )

    transaction_type: TransactionType = "other"

    transaction_date: datetime

    merchant: str | None = None

    category: str | None = None

    country: str | None = None

    channel: str | None = None

    payment_method: str | None = None

    customer_age_days: int | None = Field(
        default=None,
        ge=0,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict
    )


class BatchFraudRequest(BaseModel):
    """
    Batch fraud-analysis request.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    transaction_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=1000,
    )

    include_explanations: bool = True

    create_alerts: bool = True


class FraudInvestigationRequest(BaseModel):
    """
    Request for investigating a suspicious transaction.
    """

    model_config = ConfigDict(
        extra="forbid"
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    transaction_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
    )

    include_related_transactions: bool = True

    include_customer_history: bool = True

    include_anomalies: bool = True

    include_explanation: bool = True


# ============================================================
# Response Schemas
# ============================================================


class FraudFactor(BaseModel):
    """
    Individual factor contributing to fraud risk.
    """

    name: str

    value: Any

    contribution: float

    explanation: str


class FraudScoreResponse(BaseModel):
    """
    Fraud scoring result.
    """

    company_id: str

    transaction_id: str

    fraud_probability: float = Field(
        ge=0,
        le=1,
    )

    anomaly_score: float = Field(
        ge=0,
        le=1,
    )

    risk_score: float = Field(
        ge=0,
        le=100,
    )

    risk_level: RiskLevel

    decision: FraudDecision

    model_version: str

    threshold: float

    factors: list[FraudFactor] = Field(
        default_factory=list
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    analyzed_at: datetime


class FraudAlertResponse(BaseModel):
    """
    Fraud alert generated from a fraud detection result.
    """

    alert_id: str

    company_id: str

    transaction_id: str

    severity: RiskLevel

    title: str

    description: str

    fraud_probability: float

    risk_score: float

    status: str

    evidence: list[dict[str, Any]] = Field(
        default_factory=list
    )

    recommended_action: str

    created_at: datetime


class BatchFraudResult(BaseModel):
    """
    Batch analysis summary.
    """

    company_id: str

    total_transactions: int

    analyzed_transactions: int

    legitimate_count: int

    suspicious_count: int

    fraud_count: int

    review_count: int

    high_risk_count: int

    critical_risk_count: int

    average_risk_score: float

    results: list[FraudScoreResponse]


class FraudInvestigationResponse(BaseModel):
    """
    Detailed fraud investigation result.
    """

    company_id: str

    transaction_id: str

    fraud_score: FraudScoreResponse

    related_transactions: list[dict[str, Any]] = Field(
        default_factory=list
    )

    customer_history: list[dict[str, Any]] = Field(
        default_factory=list
    )

    anomalies: list[dict[str, Any]] = Field(
        default_factory=list
    )

    evidence: list[dict[str, Any]] = Field(
        default_factory=list
    )

    explanation: str

    recommended_actions: list[str] = Field(
        default_factory=list
    )

    investigation_confidence: float = Field(
        ge=0,
        le=1,
    )


class FraudStatisticsResponse(BaseModel):
    """
    Fraud statistics for a company and period.
    """

    company_id: str

    period_start: date

    period_end: date

    total_transactions: int

    total_transaction_value: Decimal

    suspicious_transactions: int

    confirmed_fraud_transactions: int

    fraud_rate: float

    suspicious_rate: float

    fraud_value: Decimal

    average_fraud_score: float

    high_risk_transactions: int

    critical_risk_transactions: int


class FraudModelHealthResponse(BaseModel):
    """
    Fraud model health and performance information.
    """

    status: str

    model_name: str

    model_version: str

    last_trained_at: datetime | None = None

    training_samples: int | None = None

    precision: float | None = None

    recall: float | None = None

    f1_score: float | None = None

    roc_auc: float | None = None

    drift_detected: bool

    warnings: list[str] = Field(
        default_factory=list
    )


# ============================================================
# Dependencies
# ============================================================


def get_fraud_service() -> Any:
    """
    Fraud service dependency.

    The actual service should coordinate:

        preprocessing
        feature_engineering
        supervised_model
        anomaly_model
        scoring
        thresholds
        explainability
    """

    from backend.app.fraud.fraud_service import FraudService

    return FraudService()


# ============================================================
# Security Helpers
# ============================================================


def validate_company_access(
    company_id: str,
    current_user: Any,
) -> None:
    """
    Enforce tenant/company isolation.

    Never rely exclusively on a company_id supplied by the client.
    The production implementation should use the application's
    RBAC/permissions service.
    """

    if not company_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="company_id cannot be empty.",
        )

    user_company_id = getattr(
        current_user,
        "company_id",
        None,
    )

    is_admin = getattr(
        current_user,
        "is_admin",
        False,
    )

    if (
        not is_admin
        and user_company_id
        and user_company_id != company_id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this company.",
        )


# ============================================================
# Single Transaction Analysis
# ============================================================


@router.post(
    "/score",
    response_model=FraudScoreResponse,
    summary="Score a transaction for fraud risk",
)
async def score_transaction(
    request: FraudTransactionRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_fraud_service
    ),
) -> FraudScoreResponse:

    validate_company_access(
        request.company_id,
        current_user,
    )

    try:
        result = await service.score_transaction(
            company_id=request.company_id,
            transaction_id=request.transaction_id,
            customer_id=request.customer_id,
            amount=request.amount,
            currency=request.currency.upper(),
            transaction_type=request.transaction_type,
            transaction_date=request.transaction_date,
            merchant=request.merchant,
            category=request.category,
            country=request.country,
            channel=request.channel,
            payment_method=request.payment_method,
            customer_age_days=request.customer_age_days,
            metadata=request.metadata,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unable to score transaction.",
            )

        return FraudScoreResponse(
            company_id=request.company_id,
            transaction_id=request.transaction_id,
            fraud_probability=float(
                result.get(
                    "fraud_probability",
                    0,
                )
            ),
            anomaly_score=float(
                result.get(
                    "anomaly_score",
                    0,
                )
            ),
            risk_score=float(
                result.get(
                    "risk_score",
                    0,
                )
            ),
            risk_level=result.get(
                "risk_level",
                "LOW",
            ),
            decision=result.get(
                "decision",
                "REVIEW",
            ),
            model_version=result.get(
                "model_version",
                "unknown",
            ),
            threshold=float(
                result.get(
                    "threshold",
                    0.5,
                )
            ),
            factors=[
                FraudFactor(**factor)
                for factor in result.get(
                    "factors",
                    [],
                )
            ],
            warnings=result.get(
                "warnings",
                [],
            ),
            analyzed_at=result.get(
                "analyzed_at",
                datetime.utcnow(),
            ),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        logger.warning(
            "Invalid fraud scoring request: %s",
            exc,
        )

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Fraud scoring failed "
            "company=%s transaction=%s",
            request.company_id,
            request.transaction_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to score transaction.",
        ) from exc


# ============================================================
# Batch Fraud Detection
# ============================================================


@router.post(
    "/batch",
    response_model=BatchFraudResult,
    summary="Analyze multiple transactions for fraud",
)
async def batch_fraud_detection(
    request: BatchFraudRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_fraud_service
    ),
) -> BatchFraudResult:

    validate_company_access(
        request.company_id,
        current_user,
    )

    # Remove duplicates while preserving order.
    transaction_ids = list(
        dict.fromkeys(
            request.transaction_ids
        )
    )

    try:
        result = await service.score_batch(
            company_id=request.company_id,
            transaction_ids=transaction_ids,
            include_explanations=request.include_explanations,
            create_alerts=request.create_alerts,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No transactions found for analysis.",
            )

        results = [
            FraudScoreResponse(
                **item
            )
            for item in result.get(
                "results",
                [],
            )
        ]

        return BatchFraudResult(
            company_id=request.company_id,
            total_transactions=len(
                transaction_ids
            ),
            analyzed_transactions=len(
                results
            ),
            legitimate_count=int(
                result.get(
                    "legitimate_count",
                    0,
                )
            ),
            suspicious_count=int(
                result.get(
                    "suspicious_count",
                    0,
                )
            ),
            fraud_count=int(
                result.get(
                    "fraud_count",
                    0,
                )
            ),
            review_count=int(
                result.get(
                    "review_count",
                    0,
                )
            ),
            high_risk_count=int(
                result.get(
                    "high_risk_count",
                    0,
                )
            ),
            critical_risk_count=int(
                result.get(
                    "critical_risk_count",
                    0,
                )
            ),
            average_risk_score=float(
                result.get(
                    "average_risk_score",
                    0,
                )
            ),
            results=results,
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Batch fraud detection failed "
            "company=%s",
            request.company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to analyze transactions.",
        ) from exc


# ============================================================
# Investigation
# ============================================================


@router.post(
    "/investigate",
    response_model=FraudInvestigationResponse,
    summary="Investigate a suspicious transaction",
)
async def investigate_transaction(
    request: FraudInvestigationRequest,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_fraud_service
    ),
) -> FraudInvestigationResponse:

    validate_company_access(
        request.company_id,
        current_user,
    )

    try:
        result = await service.investigate(
            company_id=request.company_id,
            transaction_id=request.transaction_id,
            include_related_transactions=(
                request.include_related_transactions
            ),
            include_customer_history=(
                request.include_customer_history
            ),
            include_anomalies=(
                request.include_anomalies
            ),
            include_explanation=(
                request.include_explanation
            ),
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Transaction investigation data not found.",
            )

        fraud_score = FraudScoreResponse(
            **result["fraud_score"]
        )

        return FraudInvestigationResponse(
            company_id=request.company_id,
            transaction_id=request.transaction_id,
            fraud_score=fraud_score,
            related_transactions=result.get(
                "related_transactions",
                [],
            ),
            customer_history=result.get(
                "customer_history",
                [],
            ),
            anomalies=result.get(
                "anomalies",
                [],
            ),
            evidence=result.get(
                "evidence",
                [],
            ),
            explanation=result.get(
                "explanation",
                "",
            ),
            recommended_actions=result.get(
                "recommended_actions",
                [],
            ),
            investigation_confidence=float(
                result.get(
                    "investigation_confidence",
                    0,
                )
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Fraud investigation failed "
            "company=%s transaction=%s",
            request.company_id,
            request.transaction_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to investigate transaction.",
        ) from exc


# ============================================================
# Fraud Alerts
# ============================================================


@router.post(
    "/alerts",
    response_model=FraudAlertResponse,
    summary="Create a fraud alert",
)
async def create_fraud_alert(
    company_id: str = Query(...),
    transaction_id: str = Query(...),
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_fraud_service
    ),
) -> FraudAlertResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    try:
        result = await service.create_alert(
            company_id=company_id,
            transaction_id=transaction_id,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Unable to create fraud alert.",
            )

        return FraudAlertResponse(
            **result
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Fraud alert creation failed "
            "company=%s transaction=%s",
            company_id,
            transaction_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to create fraud alert.",
        ) from exc


# ============================================================
# Fraud Statistics
# ============================================================


@router.get(
    "/statistics",
    response_model=FraudStatisticsResponse,
    summary="Get fraud statistics",
)
async def get_fraud_statistics(
    company_id: str,
    period_start: date,
    period_end: date,
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_fraud_service
    ),
) -> FraudStatisticsResponse:

    validate_company_access(
        company_id,
        current_user,
    )

    if period_start > period_end:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "period_start must be before "
                "period_end."
            ),
        )

    try:
        result = await service.get_statistics(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
        )

        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No fraud statistics found.",
            )

        return FraudStatisticsResponse(
            company_id=company_id,
            period_start=period_start,
            period_end=period_end,
            total_transactions=int(
                result.get(
                    "total_transactions",
                    0,
                )
            ),
            total_transaction_value=result.get(
                "total_transaction_value",
                Decimal("0"),
            ),
            suspicious_transactions=int(
                result.get(
                    "suspicious_transactions",
                    0,
                )
            ),
            confirmed_fraud_transactions=int(
                result.get(
                    "confirmed_fraud_transactions",
                    0,
                )
            ),
            fraud_rate=float(
                result.get(
                    "fraud_rate",
                    0,
                )
            ),
            suspicious_rate=float(
                result.get(
                    "suspicious_rate",
                    0,
                )
            ),
            fraud_value=result.get(
                "fraud_value",
                Decimal("0"),
            ),
            average_fraud_score=float(
                result.get(
                    "average_fraud_score",
                    0,
                )
            ),
            high_risk_transactions=int(
                result.get(
                    "high_risk_transactions",
                    0,
                )
            ),
            critical_risk_transactions=int(
                result.get(
                    "critical_risk_transactions",
                    0,
                )
            ),
        )

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception(
            "Fraud statistics failed "
            "company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve fraud statistics.",
        ) from exc


# ============================================================
# High-Risk Transactions
# ============================================================


@router.get(
    "/high-risk",
    summary="Get high-risk transactions",
)
async def get_high_risk_transactions(
    company_id: str,
    minimum_score: float = Query(
        70,
        ge=0,
        le=100,
    ),
    limit: int = Query(
        50,
        ge=1,
        le=500,
    ),
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_fraud_service
    ),
) -> dict[str, Any]:

    validate_company_access(
        company_id,
        current_user,
    )

    try:
        transactions = await service.get_high_risk_transactions(
            company_id=company_id,
            minimum_score=minimum_score,
            limit=limit,
        )

        return {
            "company_id": company_id,
            "minimum_score": minimum_score,
            "count": len(transactions),
            "transactions": transactions,
        }

    except Exception as exc:
        logger.exception(
            "High-risk transaction retrieval failed "
            "company=%s",
            company_id,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve high-risk transactions.",
        ) from exc


# ============================================================
# Model Health
# ============================================================


@router.get(
    "/model-health",
    response_model=FraudModelHealthResponse,
    summary="Get fraud model health",
)
async def fraud_model_health(
    current_user: Any = Depends(
        get_current_user
    ),
    service: Any = Depends(
        get_fraud_service
    ),
) -> FraudModelHealthResponse:

    # This endpoint should normally be restricted to
    # administrators / ML operations users.
    is_admin = getattr(
        current_user,
        "is_admin",
        False,
    )

    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Administrator access required.",
        )

    try:
        result = await service.model_health()

        return FraudModelHealthResponse(
            status=result.get(
                "status",
                "UNKNOWN",
            ),
            model_name=result.get(
                "model_name",
                "unknown",
            ),
            model_version=result.get(
                "model_version",
                "unknown",
            ),
            last_trained_at=result.get(
                "last_trained_at"
            ),
            training_samples=result.get(
                "training_samples"
            ),
            precision=result.get(
                "precision"
            ),
            recall=result.get(
                "recall"
            ),
            f1_score=result.get(
                "f1_score"
            ),
            roc_auc=result.get(
                "roc_auc"
            ),
            drift_detected=bool(
                result.get(
                    "drift_detected",
                    False,
                )
            ),
            warnings=result.get(
                "warnings",
                [],
            ),
        )

    except Exception as exc:
        logger.exception(
            "Fraud model health check failed."
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve fraud model health.",
        ) from exc


# ============================================================
# API Health
# ============================================================


@router.get(
    "/health",
    summary="Fraud API health check",
)
async def fraud_api_health() -> dict[str, str]:

    return {
        "service": "fraud-api",
        "status": "healthy",
    }