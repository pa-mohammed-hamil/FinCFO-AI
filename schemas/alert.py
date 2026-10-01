"""
FinCo AI - Alert Schemas

File:
    backend/app/schemas/alert.py

Purpose:
    Pydantic schemas for the FinCo AI alert API layer.

Responsibilities:
    - Validate alert requests
    - Define alert response contracts
    - Standardize alert severity/status/category
    - Support alert acknowledgement
    - Support alert resolution
    - Support alert filtering
    - Support alert escalation
    - Support structured evidence
    - Support audit-friendly alert data

Architecture:

    Alert Engine
         ↓
    Alert Model / Service
         ↓
    Alert Schemas
         ↓
    FastAPI Routes
         ↓
    Frontend / Agent / Power BI

Important:
    This module contains validation and API contracts only.
    It does NOT decide whether an alert should be created.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


# ============================================================================
# Base Configuration
# ============================================================================

class AlertSchemaBase(BaseModel):
    """
    Base Pydantic configuration shared by alert schemas.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
        use_enum_values=True,
    )


# ============================================================================
# Alert Enums
# ============================================================================

class AlertSeverity(str, Enum):
    """
    Severity of a financial alert.
    """

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """
    Alert lifecycle status.
    """

    OPEN = "open"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    DISMISSED = "dismissed"


class AlertCategory(str, Enum):
    """
    High-level alert classification.
    """

    FINANCIAL = "financial"
    REVENUE = "revenue"
    EXPENSE = "expense"
    PROFITABILITY = "profitability"
    CASH_FLOW = "cash_flow"
    LIQUIDITY = "liquidity"
    FORECAST = "forecast"
    FRAUD = "fraud"
    RISK = "risk"
    DEBT = "debt"
    BUDGET = "budget"
    TRANSACTION = "transaction"
    DOCUMENT = "document"
    SYSTEM = "system"


class AlertType(str, Enum):
    """
    Specific alert event types.
    """

    FINANCIAL_PROFILE_DOWN = (
        "financial_profile_down"
    )

    PREVIOUS_PERIOD_DECLINE = (
        "previous_period_decline"
    )

    SEVERE_LOSS_RISK = (
        "severe_loss_risk"
    )

    REVENUE_DECLINE = (
        "revenue_decline"
    )

    EXPENSE_SPIKE = (
        "expense_spike"
    )

    PROFIT_MARGIN_DROP = (
        "profit_margin_drop"
    )

    CASH_FLOW_DECLINE = (
        "cash_flow_decline"
    )

    LIQUIDITY_RISK = (
        "liquidity_risk"
    )

    DEBT_RISK = (
        "debt_risk"
    )

    FORECAST_DOWNGRADE = (
        "forecast_downgrade"
    )

    FORECAST_LOSS_RISK = (
        "forecast_loss_risk"
    )

    FRAUD_SUSPECTED = (
        "fraud_suspected"
    )

    ANOMALY_DETECTED = (
        "anomaly_detected"
    )

    BUDGET_VARIANCE = (
        "budget_variance"
    )

    TRANSACTION_RISK = (
        "transaction_risk"
    )

    DOCUMENT_VALIDATION = (
        "document_validation"
    )

    SYSTEM_ERROR = (
        "system_error"
    )

    CUSTOM = "custom"


class AlertSource(str, Enum):
    """
    System that generated the alert.
    """

    RULE_ENGINE = "rule_engine"
    ML_MODEL = "ml_model"
    FORECASTING = "forecasting"
    FRAUD_ENGINE = "fraud_engine"
    FINANCIAL_ENGINE = "financial_engine"
    ANOMALY_DETECTION = "anomaly_detection"
    AGENT = "agent"
    USER = "user"
    SYSTEM = "system"


class AlertPriority(str, Enum):
    """
    Operational handling priority.
    """

    P0 = "p0"
    P1 = "p1"
    P2 = "p2"
    P3 = "p3"


class EscalationLevel(str, Enum):
    """
    Escalation level for alert handling.
    """

    NONE = "none"
    TEAM = "team"
    MANAGER = "manager"
    EXECUTIVE = "executive"
    CRITICAL_RESPONSE = "critical_response"


# ============================================================================
# Evidence Schema
# ============================================================================

class AlertEvidence(AlertSchemaBase):
    """
    Structured evidence supporting an alert.

    Example:

        metric_name = "net_profit"
        observed_value = -25000
        threshold = 0
    """

    metric_name: str = Field(
        ...,
        min_length=1,
        max_length=150,
        description=(
            "Financial or operational metric "
            "associated with the alert."
        ),
    )

    metric_label: Optional[str] = Field(
        default=None,
        max_length=200,
    )

    observed_value: Optional[float] = Field(
        default=None,
        description="Observed metric value.",
    )

    baseline_value: Optional[float] = Field(
        default=None,
        description="Baseline or previous value.",
    )

    threshold: Optional[float] = Field(
        default=None,
        description="Alert threshold.",
    )

    unit: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    change_pct: Optional[float] = Field(
        default=None,
        description="Percentage change from baseline.",
    )

    direction: Optional[str] = Field(
        default=None,
        max_length=30,
    )

    source: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    source_reference: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    period: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Alert Context
# ============================================================================

class AlertContext(AlertSchemaBase):
    """
    Context surrounding the alert.
    """

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    company_name: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    fiscal_year: Optional[int] = Field(
        default=None,
        ge=1900,
        le=2200,
    )

    fiscal_period: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    currency: Optional[str] = Field(
        default="USD",
        min_length=3,
        max_length=3,
    )

    department: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    business_unit: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    region: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("currency")
    @classmethod
    def validate_currency(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return value

        return value.upper()


# ============================================================================
# Alert Create Schema
# ============================================================================

class AlertCreate(AlertSchemaBase):
    """
    Schema used when creating a new alert.
    """

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    alert_type: AlertType = Field(
        ...,
        description="Specific alert event type.",
    )

    category: AlertCategory = Field(
        ...,
        description="High-level alert category.",
    )

    severity: AlertSeverity = Field(
        ...,
        description="Alert severity.",
    )

    priority: AlertPriority = Field(
        default=AlertPriority.P2,
    )

    title: str = Field(
        ...,
        min_length=3,
        max_length=255,
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=10000,
    )

    source: AlertSource = Field(
        default=AlertSource.RULE_ENGINE,
    )

    metric_name: Optional[str] = Field(
        default=None,
        max_length=150,
    )

    observed_value: Optional[float] = Field(
        default=None,
    )

    threshold_value: Optional[float] = Field(
        default=None,
    )

    baseline_value: Optional[float] = Field(
        default=None,
    )

    change_pct: Optional[float] = Field(
        default=None,
    )

    risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    evidence: list[AlertEvidence] = Field(
        default_factory=list,
        max_length=50,
    )

    context: Optional[AlertContext] = None

    tags: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )

    @field_validator("title")
    @classmethod
    def validate_title(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Alert title cannot be empty."
            )

        return value

    @field_validator("message")
    @classmethod
    def validate_message(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Alert message cannot be empty."
            )

        return value

    @field_validator("tags")
    @classmethod
    def validate_tags(
        cls,
        value: list[str],
    ) -> list[str]:

        cleaned = []

        for tag in value:

            tag = tag.strip().lower()

            if tag and tag not in cleaned:
                cleaned.append(tag)

        return cleaned


# ============================================================================
# Alert Update
# ============================================================================

class AlertUpdate(AlertSchemaBase):
    """
    Schema for partially updating an alert.
    """

    severity: Optional[AlertSeverity] = None

    priority: Optional[AlertPriority] = None

    status: Optional[AlertStatus] = None

    title: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=255,
    )

    message: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=5000,
    )

    assigned_to: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    escalation_level: Optional[
        EscalationLevel
    ] = None

    tags: Optional[list[str]] = Field(
        default=None,
        max_length=30,
    )

    metadata: Optional[
        dict[str, Any]
    ] = None

    @field_validator("title")
    @classmethod
    def validate_title(
        cls,
        value: Optional[str],
    ) -> Optional[str]:

        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "Alert title cannot be empty."
            )

        return value

    @field_validator("message")
    @classmethod
    def validate_message(
        cls,
        value: Optional[str],
    ) -> Optional[str]:

        if value is None:
            return None

        value = value.strip()

        if not value:
            raise ValueError(
                "Alert message cannot be empty."
            )

        return value


# ============================================================================
# Alert Acknowledge
# ============================================================================

class AlertAcknowledgeRequest(AlertSchemaBase):
    """
    Request to acknowledge an alert.
    """

    acknowledged_by: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    note: Optional[str] = Field(
        default=None,
        max_length=2000,
    )


# ============================================================================
# Alert Resolve
# ============================================================================

class AlertResolveRequest(AlertSchemaBase):
    """
    Request to resolve an alert.
    """

    resolved_by: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    resolution_note: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    resolution_code: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    @field_validator("resolution_note")
    @classmethod
    def validate_resolution_note(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Resolution note cannot be empty."
            )

        return value


# ============================================================================
# Alert Dismiss
# ============================================================================

class AlertDismissRequest(AlertSchemaBase):
    """
    Request to dismiss an alert.
    """

    dismissed_by: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    reason: str = Field(
        ...,
        min_length=1,
        max_length=3000,
    )

    @field_validator("reason")
    @classmethod
    def validate_reason(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Dismissal reason cannot be empty."
            )

        return value


# ============================================================================
# Alert Escalation
# ============================================================================

class AlertEscalateRequest(AlertSchemaBase):
    """
    Request to escalate an alert.
    """

    escalated_by: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    escalation_level: EscalationLevel

    assigned_to: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    reason: str = Field(
        ...,
        min_length=1,
        max_length=3000,
    )

    @field_validator("reason")
    @classmethod
    def validate_reason(
        cls,
        value: str,
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Escalation reason cannot be empty."
            )

        return value


# ============================================================================
# Alert Response
# ============================================================================

class AlertResponse(AlertSchemaBase):
    """
    Complete alert returned by the API.
    """

    id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    alert_type: AlertType

    category: AlertCategory

    severity: AlertSeverity

    priority: AlertPriority

    status: AlertStatus

    title: str

    message: str

    description: Optional[str] = None

    source: AlertSource

    metric_name: Optional[str] = None

    observed_value: Optional[float] = None

    threshold_value: Optional[float] = None

    baseline_value: Optional[float] = None

    change_pct: Optional[float] = None

    risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    evidence: list[AlertEvidence] = Field(
        default_factory=list,
    )

    context: Optional[AlertContext] = None

    assigned_to: Optional[str] = None

    escalation_level: EscalationLevel = (
        EscalationLevel.NONE
    )

    acknowledged_by: Optional[str] = None

    acknowledged_at: Optional[datetime] = None

    resolved_by: Optional[str] = None

    resolved_at: Optional[datetime] = None

    resolution_note: Optional[str] = None

    dismissed_by: Optional[str] = None

    dismissed_at: Optional[datetime] = None

    dismissal_reason: Optional[str] = None

    created_at: datetime

    updated_at: datetime

    tags: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Alert List Item
# ============================================================================

class AlertListItem(AlertSchemaBase):
    """
    Lightweight alert representation for dashboards.
    """

    id: str

    company_id: str

    alert_type: AlertType

    category: AlertCategory

    severity: AlertSeverity

    priority: AlertPriority

    status: AlertStatus

    title: str

    message: str

    risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    created_at: datetime

    updated_at: datetime


# ============================================================================
# Alert List Response
# ============================================================================

class AlertListResponse(AlertSchemaBase):
    """
    Paginated alert response.
    """

    items: list[AlertListItem]

    total: int = Field(
        ...,
        ge=0,
    )

    page: int = Field(
        ...,
        ge=1,
    )

    page_size: int = Field(
        ...,
        ge=1,
        le=100,
    )

    total_pages: int = Field(
        ...,
        ge=0,
    )

    has_next: bool

    has_previous: bool


# ============================================================================
# Alert Filter
# ============================================================================

class AlertFilter(AlertSchemaBase):
    """
    Filtering parameters for alert queries.
    """

    company_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    category: Optional[
        AlertCategory
    ] = None

    alert_type: Optional[
        AlertType
    ] = None

    severity: Optional[
        AlertSeverity
    ] = None

    priority: Optional[
        AlertPriority
    ] = None

    status: Optional[
        AlertStatus
    ] = None

    source: Optional[
        AlertSource
    ] = None

    assigned_to: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    escalation_level: Optional[
        EscalationLevel
    ] = None

    min_risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    max_risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    start_date: Optional[datetime] = None

    end_date: Optional[datetime] = None

    tags: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    search: Optional[str] = Field(
        default=None,
        max_length=255,
    )

    page: int = Field(
        default=1,
        ge=1,
    )

    page_size: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    sort_by: str = Field(
        default="created_at",
        max_length=50,
    )

    sort_order: str = Field(
        default="desc",
        max_length=4,
    )

    @field_validator("sort_order")
    @classmethod
    def validate_sort_order(
        cls,
        value: str,
    ) -> str:

        value = value.lower()

        if value not in {
            "asc",
            "desc",
        }:
            raise ValueError(
                "sort_order must be 'asc' or 'desc'."
            )

        return value

    @model_validator(mode="after")
    def validate_filter_range(self):

        if (
            self.min_risk_score is not None
            and self.max_risk_score is not None
            and self.min_risk_score
            > self.max_risk_score
        ):
            raise ValueError(
                "min_risk_score cannot exceed "
                "max_risk_score."
            )

        if (
            self.start_date is not None
            and self.end_date is not None
            and self.start_date > self.end_date
        ):
            raise ValueError(
                "start_date cannot be later "
                "than end_date."
            )

        return self


# ============================================================================
# Alert Statistics
# ============================================================================

class AlertStatistics(AlertSchemaBase):
    """
    Aggregated alert statistics for dashboard use.
    """

    total: int = Field(
        default=0,
        ge=0,
    )

    open: int = Field(
        default=0,
        ge=0,
    )

    acknowledged: int = Field(
        default=0,
        ge=0,
    )

    investigating: int = Field(
        default=0,
        ge=0,
    )

    escalated: int = Field(
        default=0,
        ge=0,
    )

    resolved: int = Field(
        default=0,
        ge=0,
    )

    dismissed: int = Field(
        default=0,
        ge=0,
    )

    info: int = Field(
        default=0,
        ge=0,
    )

    low: int = Field(
        default=0,
        ge=0,
    )

    medium: int = Field(
        default=0,
        ge=0,
    )

    high: int = Field(
        default=0,
        ge=0,
    )

    critical: int = Field(
        default=0,
        ge=0,
    )

    average_risk_score: Optional[float] = (
        Field(
            default=None,
            ge=0,
            le=100,
        )
    )

    critical_rate_pct: float = Field(
        default=0.0,
        ge=0,
        le=100,
    )


# ============================================================================
# Alert Rule Schema
# ============================================================================

class AlertRuleCondition(AlertSchemaBase):
    """
    A single deterministic alert-rule condition.

    Example:

        metric = "net_profit"
        operator = "lt"
        threshold = 0
    """

    metric: str = Field(
        ...,
        min_length=1,
        max_length=150,
    )

    operator: str = Field(
        ...,
        min_length=1,
        max_length=10,
    )

    threshold: float

    unit: Optional[str] = Field(
        default=None,
        max_length=50,
    )

    @field_validator("operator")
    @classmethod
    def validate_operator(
        cls,
        value: str,
    ) -> str:

        value = value.lower()

        allowed = {
            "gt",
            "gte",
            "lt",
            "lte",
            "eq",
            "neq",
        }

        if value not in allowed:
            raise ValueError(
                "Unsupported operator. "
                f"Allowed: {sorted(allowed)}"
            )

        return value


class AlertRule(AlertSchemaBase):
    """
    Configuration contract for deterministic alert rules.
    """

    id: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    name: str = Field(
        ...,
        min_length=3,
        max_length=255,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=3000,
    )

    alert_type: AlertType

    category: AlertCategory

    severity: AlertSeverity

    priority: AlertPriority = (
        AlertPriority.P2
    )

    source: AlertSource = (
        AlertSource.RULE_ENGINE
    )

    conditions: list[
        AlertRuleCondition
    ] = Field(
        ...,
        min_length=1,
        max_length=20,
    )

    enabled: bool = True

    cooldown_minutes: int = Field(
        default=60,
        ge=0,
        le=10080,
    )

    escalation_level: EscalationLevel = (
        EscalationLevel.NONE
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Alert Event
# ============================================================================

class AlertEvent(AlertSchemaBase):
    """
    Internal/event-bus representation of an alert event.

    Useful for:

        Alert Engine
             ↓
        Event Bus
             ↓
        Supervisor Agent
             ↓
        Notification
             ↓
        Audit Log
    """

    event_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    alert_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    company_id: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    event_type: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    alert_type: AlertType

    severity: AlertSeverity

    occurred_at: datetime

    payload: dict[str, Any] = Field(
        default_factory=dict,
    )

    correlation_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    causation_id: Optional[str] = Field(
        default=None,
        max_length=100,
    )


# ============================================================================
# Agent Investigation Context
# ============================================================================

class AlertInvestigationContext(
    AlertSchemaBase
):
    """
    Structured context passed from the alert engine
    to the Supervisor Agent.

    The agent receives facts and evidence rather than
    being allowed to invent the existence of an alert.
    """

    alert_id: str

    company_id: str

    alert_type: AlertType

    category: AlertCategory

    severity: AlertSeverity

    risk_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )

    title: str

    message: str

    evidence: list[AlertEvidence] = Field(
        default_factory=list,
    )

    baseline_metrics: dict[
        str,
        float
    ] = Field(
        default_factory=dict,
    )

    current_metrics: dict[
        str,
        float
    ] = Field(
        default_factory=dict,
    )

    forecast_metrics: dict[
        str,
        float
    ] = Field(
        default_factory=dict,
    )

    related_alert_ids: list[str] = Field(
        default_factory=list,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Agent Investigation Result
# ============================================================================

class AlertInvestigationResult(
    AlertSchemaBase
):
    """
    Structured result produced by the investigation
    workflow.

    This allows the agent layer to return structured
    facts/recommendations while the alert engine remains
    deterministic.
    """

    alert_id: str

    root_cause: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    contributing_factors: list[str] = Field(
        default_factory=list,
        max_length=30,
    )

    evidence: list[AlertEvidence] = Field(
        default_factory=list,
        max_length=50,
    )

    forecast_impact: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    recommended_actions: list[str] = Field(
        default_factory=list,
        max_length=20,
    )

    confidence: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    requires_human_review: bool = True

    human_review_reason: Optional[str] = Field(
        default=None,
        max_length=3000,
    )

    citations: list[str] = Field(
        default_factory=list,
        max_length=50,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Alert Action
# ============================================================================

class AlertActionType(str, Enum):
    """
    Supported alert actions.
    """

    ACKNOWLEDGE = "acknowledge"
    INVESTIGATE = "investigate"
    ESCALATE = "escalate"
    RESOLVE = "resolve"
    DISMISS = "dismiss"
    REOPEN = "reopen"


class AlertActionRequest(AlertSchemaBase):
    """
    Generic alert action request.
    """

    action: AlertActionType

    performed_by: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    note: Optional[str] = Field(
        default=None,
        max_length=3000,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Alert Action Response
# ============================================================================

class AlertActionResponse(AlertSchemaBase):
    """
    Response after an alert action.
    """

    success: bool

    alert_id: str

    action: AlertActionType

    status: AlertStatus

    message: str

    performed_by: str

    performed_at: datetime

    audit_log_id: Optional[str] = None


# ============================================================================
# Notification Schema
# ============================================================================

class AlertNotificationChannel(str, Enum):
    """
    Supported notification channels.
    """

    IN_APP = "in_app"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"


class AlertNotificationRequest(
    AlertSchemaBase
):
    """
    Notification request for an alert.
    """

    alert_id: str

    channels: list[
        AlertNotificationChannel
    ] = Field(
        ...,
        min_length=1,
        max_length=10,
    )

    recipients: list[str] = Field(
        default_factory=list,
        max_length=100,
    )

    message_override: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
    )


# ============================================================================
# Alert Dashboard Summary
# ============================================================================

class AlertDashboardSummary(
    AlertSchemaBase
):
    """
    High-level dashboard payload.
    """

    company_id: str

    statistics: AlertStatistics

    latest_alerts: list[
        AlertListItem
    ] = Field(
        default_factory=list,
        max_length=20,
    )

    top_risk_alerts: list[
        AlertListItem
    ] = Field(
        default_factory=list,
        max_length=20,
    )

    categories: dict[
        str,
        int
    ] = Field(
        default_factory=dict,
    )

    severity_distribution: dict[
        str,
        int
    ] = Field(
        default_factory=dict,
    )

    generated_at: datetime


# ============================================================================
# Alert Bulk Action
# ============================================================================

class AlertBulkActionRequest(
    AlertSchemaBase
):
    """
    Apply an action to multiple alerts.
    """

    alert_ids: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    action: AlertActionType

    performed_by: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )

    note: Optional[str] = Field(
        default=None,
        max_length=3000,
    )

    @field_validator("alert_ids")
    @classmethod
    def validate_alert_ids(
        cls,
        value: list[str],
    ) -> list[str]:

        cleaned = []

        for alert_id in value:

            alert_id = alert_id.strip()

            if (
                alert_id
                and alert_id not in cleaned
            ):
                cleaned.append(alert_id)

        if not cleaned:
            raise ValueError(
                "At least one valid alert ID "
                "is required."
            )

        return cleaned


class AlertBulkActionResponse(
    AlertSchemaBase
):
    """
    Result of a bulk alert operation.
    """

    success: bool

    action: AlertActionType

    requested_count: int

    successful_count: int

    failed_count: int

    successful_alert_ids: list[str] = Field(
        default_factory=list,
    )

    failed_alert_ids: list[str] = Field(
        default_factory=list,
    )

    errors: dict[
        str,
        str
    ] = Field(
        default_factory=dict,
    )

    performed_by: str

    performed_at: datetime


# ============================================================================
# Exports
# ============================================================================

__all__ = [
    "AlertSchemaBase",
    "AlertSeverity",
    "AlertStatus",
    "AlertCategory",
    "AlertType",
    "AlertSource",
    "AlertPriority",
    "EscalationLevel",
    "AlertEvidence",
    "AlertContext",
    "AlertCreate",
    "AlertUpdate",
    "AlertResponse",
    "AlertListItem",
    "AlertListResponse",
    "AlertFilter",
    "AlertStatistics",
    "AlertRuleCondition",
    "AlertRule",
    "AlertEvent",
    "AlertInvestigationContext",
    "AlertInvestigationResult",
    "AlertActionType",
    "AlertActionRequest",
    "AlertActionResponse",
    "AlertNotificationChannel",
    "AlertNotificationRequest",
    "AlertDashboardSummary",
    "AlertBulkActionRequest",
    "AlertBulkActionResponse",
]