"""
FinCo AI - Fraud Schemas

Pydantic schemas for:

- Fraud detection
- Anomaly detection
- Fraud scoring
- Risk classification
- Fraud investigation
- Explainability
- Transaction analysis
- Fraud alerts
- Fraud dashboard
- Model evaluation
"""

from datetime import date, datetime
from decimal import Decimal
from enum import Enum
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


# ============================================================
# ENUMS
# ============================================================

class FraudType(str, Enum):
    """Types of potential financial fraud."""

    PAYMENT_FRAUD = "payment_fraud"
    TRANSACTION_FRAUD = "transaction_fraud"
    INVOICE_FRAUD = "invoice_fraud"
    ACCOUNT_TAKEOVER = "account_takeover"
    DUPLICATE_TRANSACTION = "duplicate_transaction"
    UNUSUAL_TRANSACTION = "unusual_transaction"
    VENDOR_FRAUD = "vendor_fraud"
    EXPENSE_FRAUD = "expense_fraud"
    INTERNAL_FRAUD = "internal_fraud"
    OTHER = "other"


class FraudRiskLevel(str, Enum):
    """Fraud risk levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class FraudDecision(str, Enum):
    """Final fraud assessment."""

    LEGITIMATE = "legitimate"
    SUSPICIOUS = "suspicious"
    FRAUD = "fraud"
    REVIEW = "review"


class DetectionMethod(str, Enum):
    """Fraud detection techniques."""

    RULE_BASED = "rule_based"
    SUPERVISED = "supervised"
    ANOMALY_DETECTION = "anomaly_detection"
    HYBRID = "hybrid"


class InvestigationStatus(str, Enum):
    """Fraud investigation status."""

    OPEN = "open"
    IN_REVIEW = "in_review"
    ESCALATED = "escalated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


# ============================================================
# FRAUD BASE
# ============================================================

class FraudBase(BaseModel):
    """Common fraud fields."""

    model_config = ConfigDict(
        use_enum_values=True,
        extra="forbid",
    )

    company_id: int = Field(
        ...,
        gt=0,
    )

    transaction_id: Optional[int] = Field(
        default=None,
        gt=0,
    )


# ============================================================
# FRAUD DETECTION REQUEST
# ============================================================

class FraudDetectionRequest(FraudBase):
    """Request to analyze a transaction for fraud."""

    amount: Decimal = Field(
        ...,
        ge=0,
    )

    transaction_date: datetime

    customer_id: Optional[int] = None

    supplier_id: Optional[int] = None

    payment_method: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
    )

    country: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    ip_address: Optional[str] = None

    device_id: Optional[str] = None

    metadata: Dict[str, object] = Field(
        default_factory=dict,
    )

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper()


# ============================================================
# FRAUD SCORE
# ============================================================

class FraudScore(BaseModel):
    """Fraud probability and risk score."""

    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
    )

    risk_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
    )

    risk_level: FraudRiskLevel

    decision: FraudDecision


# ============================================================
# ANOMALY RESULT
# ============================================================

class AnomalyResult(BaseModel):
    """Anomaly detection result."""

    is_anomaly: bool

    anomaly_score: float

    threshold: Optional[float] = None

    anomaly_type: Optional[str] = None

    explanation: Optional[str] = None


# ============================================================
# RULE RESULT
# ============================================================

class FraudRuleResult(BaseModel):
    """Individual fraud rule result."""

    rule_id: str

    rule_name: str

    triggered: bool

    severity: FraudRiskLevel

    score: float = Field(
        default=0.0,
        ge=0,
        le=100,
    )

    reason: Optional[str] = None

    evidence: List[str] = Field(
        default_factory=list,
    )


# ============================================================
# FRAUD EXPLANATION
# ============================================================

class FraudExplanation(BaseModel):
    """Explainable fraud detection result."""

    summary: str

    key_factors: List[str] = Field(
        default_factory=list,
    )

    positive_signals: List[str] = Field(
        default_factory=list,
    )

    suspicious_signals: List[str] = Field(
        default_factory=list,
    )

    triggered_rules: List[FraudRuleResult] = Field(
        default_factory=list,
    )

    model_reasoning: Optional[str] = None

    recommended_action: Optional[str] = None


# ============================================================
# FRAUD DETECTION RESPONSE
# ============================================================

class FraudDetectionResponse(FraudBase):
    """Complete fraud detection result."""

    id: Optional[int] = None

    amount: Decimal

    currency: str = "USD"

    detection_method: DetectionMethod

    score: FraudScore

    anomaly: Optional[AnomalyResult] = None

    explanation: Optional[FraudExplanation] = None

    created_at: datetime


# ============================================================
# FRAUD ALERT
# ============================================================

class FraudAlertCreate(FraudBase):
    """Create a fraud alert."""

    fraud_type: FraudType

    risk_level: FraudRiskLevel

    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    title: str = Field(
        ...,
        min_length=3,
        max_length=300,
    )

    description: str = Field(
        ...,
        min_length=1,
        max_length=3000,
    )

    evidence: List[str] = Field(
        default_factory=list,
    )

    recommended_action: Optional[str] = None


class FraudAlertResponse(FraudAlertCreate):
    """Fraud alert API response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    status: InvestigationStatus

    created_at: datetime

    updated_at: Optional[datetime] = None


# ============================================================
# FRAUD INVESTIGATION
# ============================================================

class FraudInvestigationCreate(BaseModel):
    """Start a fraud investigation."""

    fraud_alert_id: int = Field(
        ...,
        gt=0,
    )

    priority: FraudRiskLevel = FraudRiskLevel.MEDIUM

    notes: Optional[str] = Field(
        default=None,
        max_length=5000,
    )


class FraudInvestigationUpdate(BaseModel):
    """Update an investigation."""

    status: InvestigationStatus

    notes: Optional[str] = Field(
        default=None,
        max_length=5000,
    )

    final_decision: Optional[FraudDecision] = None


class FraudInvestigationResponse(BaseModel):
    """Fraud investigation response."""

    model_config = ConfigDict(
        from_attributes=True,
        use_enum_values=True,
    )

    id: int

    fraud_alert_id: int

    status: InvestigationStatus

    priority: FraudRiskLevel

    assigned_to: Optional[int] = None

    notes: Optional[str] = None

    final_decision: Optional[FraudDecision] = None

    created_at: datetime

    updated_at: Optional[datetime] = None

    resolved_at: Optional[datetime] = None


# ============================================================
# TRANSACTION RISK
# ============================================================

class TransactionRisk(BaseModel):
    """Risk analysis for a transaction."""

    transaction_id: int

    amount: Decimal

    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    risk_level: FraudRiskLevel

    fraud_probability: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    anomaly_score: Optional[float] = None

    unusual_amount: bool = False

    unusual_frequency: bool = False

    unusual_location: bool = False

    unusual_time: bool = False

    duplicate_detected: bool = False

    vendor_risk: Optional[float] = None

    customer_risk: Optional[float] = None

    reasons: List[str] = Field(
        default_factory=list,
    )


# ============================================================
# FRAUD BATCH REQUEST
# ============================================================

class FraudBatchRequest(BaseModel):
    """Request to analyze multiple transactions."""

    company_id: int = Field(
        ...,
        gt=0,
    )

    transaction_ids: List[int] = Field(
        ...,
        min_length=1,
        max_length=10000,
    )

    detection_method: DetectionMethod = DetectionMethod.HYBRID

    create_alerts: bool = True


class FraudBatchResponse(BaseModel):
    """Batch fraud analysis result."""

    company_id: int

    total_transactions: int

    analyzed_transactions: int

    suspicious_transactions: int

    fraudulent_transactions: int

    alerts_created: int

    results: List[TransactionRisk] = Field(
        default_factory=list,
    )


# ============================================================
# FRAUD MODEL CONFIGURATION
# ============================================================

class FraudModelConfig(BaseModel):
    """Fraud ML model configuration."""

    model_name: str

    model_version: Optional[str] = None

    model_type: DetectionMethod

    threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
    )

    features: List[str] = Field(
        default_factory=list,
    )

    parameters: Dict[str, object] = Field(
        default_factory=dict,
    )


# ============================================================
# FRAUD MODEL METRICS
# ============================================================

class FraudModelMetrics(BaseModel):
    """Fraud model evaluation metrics."""

    accuracy: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    precision: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    recall: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    f1_score: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    roc_auc: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    pr_auc: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    false_positive_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )

    false_negative_rate: Optional[float] = Field(
        default=None,
        ge=0,
        le=1,
    )


# ============================================================
# FRAUD MODEL EVALUATION
# ============================================================

class FraudModelEvaluation(BaseModel):
    """Complete fraud model evaluation."""

    model: FraudModelConfig

    metrics: FraudModelMetrics

    training_samples: int = 0

    fraud_samples: int = 0

    legitimate_samples: int = 0

    evaluation_date: datetime

    notes: Optional[str] = None


# ============================================================
# FRAUD SUMMARY
# ============================================================

class FraudSummary(BaseModel):
    """Executive fraud summary."""

    company_id: int

    analysis_date: date

    total_transactions: int = 0

    analyzed_transactions: int = 0

    suspicious_transactions: int = 0

    fraudulent_transactions: int = 0

    total_at_risk_amount: Decimal = Decimal("0")

    confirmed_fraud_amount: Decimal = Decimal("0")

    average_risk_score: Optional[float] = None

    highest_risk_score: Optional[float] = None

    critical_alerts: int = 0

    high_alerts: int = 0


# ============================================================
# FRAUD TREND
# ============================================================

class FraudTrendPoint(BaseModel):
    """Fraud trend for a period."""

    period: str

    transaction_count: int = 0

    suspicious_count: int = 0

    fraud_count: int = 0

    fraud_amount: Decimal = Decimal("0")

    average_risk_score: Optional[float] = None


class FraudTrendResponse(BaseModel):
    """Fraud trend analysis."""

    company_id: int

    points: List[FraudTrendPoint] = Field(
        default_factory=list,
    )

    trend: Optional[str] = None

    explanation: Optional[str] = None


# ============================================================
# FRAUD DASHBOARD
# ============================================================

class FraudDashboardResponse(BaseModel):
    """Executive fraud dashboard."""

    company_id: int

    summary: FraudSummary

    trends: Optional[FraudTrendResponse] = None

    high_risk_transactions: List[TransactionRisk] = Field(
        default_factory=list,
    )

    active_alerts: List[FraudAlertResponse] = Field(
        default_factory=list,
    )

    model_metrics: Optional[FraudModelMetrics] = None

    generated_at: datetime


# ============================================================
# FRAUD INVESTIGATION ANALYSIS
# ============================================================

class FraudInvestigationAnalysis(BaseModel):
    """AI-assisted fraud investigation."""

    fraud_alert_id: int

    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
    )

    decision: FraudDecision

    evidence: List[str] = Field(
        default_factory=list,
    )

    related_transactions: List[int] = Field(
        default_factory=list,
    )

    suspicious_patterns: List[str] = Field(
        default_factory=list,
    )

    root_causes: List[str] = Field(
        default_factory=list,
    )

    rag_evidence: List[str] = Field(
        default_factory=list,
    )

    recommended_actions: List[str] = Field(
        default_factory=list,
    )

    human_review_required: bool = False