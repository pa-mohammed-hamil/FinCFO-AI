from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# Company Base
# ============================================================

class CompanyBase(BaseModel):
    """Common company fields."""

    name: str = Field(
        ...,
        min_length=2,
        max_length=200,
    )

    legal_name: Optional[str] = Field(
        default=None,
        max_length=250,
    )

    industry: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    country: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    currency: str = Field(
        default="USD",
        min_length=3,
        max_length=3,
    )

    fiscal_year_end: Optional[str] = Field(
        default=None,
        max_length=20,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=2000,
    )


# ============================================================
# Create Company
# ============================================================

class CompanyCreate(CompanyBase):
    """Schema for creating a company."""

    pass


# ============================================================
# Update Company
# ============================================================

class CompanyUpdate(BaseModel):
    """Schema for updating company information."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=200,
    )

    legal_name: Optional[str] = Field(
        default=None,
        max_length=250,
    )

    industry: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    country: Optional[str] = Field(
        default=None,
        max_length=100,
    )

    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    fiscal_year_end: Optional[str] = Field(
        default=None,
        max_length=20,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=2000,
    )


# ============================================================
# Company Response
# ============================================================

class CompanyResponse(CompanyBase):
    """Public company response."""

    model_config = ConfigDict(from_attributes=True)

    id: int

    is_active: bool = True

    created_at: datetime
    updated_at: Optional[datetime] = None


# ============================================================
# Company Summary
# ============================================================

class CompanySummary(BaseModel):
    """Compact company information for dashboards."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    industry: Optional[str] = None
    country: Optional[str] = None
    currency: str

    is_active: bool = True


# ============================================================
# Company Financial Profile
# ============================================================

class CompanyFinancialProfile(BaseModel):
    """Financial configuration used by FinCo AI."""

    company_id: int

    currency: str = "USD"

    fiscal_year_end: Optional[str] = None

    reporting_frequency: str = Field(
        default="monthly",
        max_length=20,
    )

    revenue_threshold: Optional[float] = None

    expense_threshold: Optional[float] = None

    loss_threshold: Optional[float] = None


# ============================================================
# Company Statistics
# ============================================================

class CompanyStatistics(BaseModel):
    """High-level company statistics for the dashboard."""

    company_id: int

    total_documents: int = 0
    total_transactions: int = 0

    total_revenue: float = 0.0
    total_expenses: float = 0.0
    net_profit: float = 0.0

    financial_health_score: Optional[float] = None
    financial_risk_score: Optional[float] = None

    active_alerts: int = 0
    critical_alerts: int = 0


# ============================================================
# Company List Response
# ============================================================

class CompanyListResponse(BaseModel):
    """Paginated company listing."""

    items: list[CompanyResponse]

    total: int

    page: int = 1
    page_size: int = 20


# ============================================================
# Company Status
# ============================================================

class CompanyStatusUpdate(BaseModel):
    """Activate or deactivate a company."""

    is_active: bool


# ============================================================
# Company Dashboard
# ============================================================

class CompanyDashboardResponse(BaseModel):
    """Combined company information for the executive dashboard."""

    company: CompanyResponse

    statistics: CompanyStatistics

    financial_profile: Optional[CompanyFinancialProfile] = None