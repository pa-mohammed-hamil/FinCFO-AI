"""
FinCo AI - Company API
Path: backend/app/api/company.py

Responsibilities
----------------
- Create companies
- Retrieve company information
- Update company information
- List companies available to the authenticated user
- Company membership / access checks
- Company-level statistics
- Company activation/deactivation

Architecture
------------
Frontend
    ↓
Company API
    ↓
Company Service / Repository
    ↓
PostgreSQL

Company context is used by:
    Financial Analysis
    Forecasting
    Fraud Detection
    Risk Engine
    RAG
    What-If Analysis
    Recommendations
    Reports
    Alerts
    Audit Logs
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)

from app.api.auth import (
    AuthUser,
    UserRole,
    get_current_user,
    require_roles,
)


# ============================================================================
# ROUTER
# ============================================================================

router = APIRouter(
    prefix="/companies",
    tags=["Companies"],
)


# ============================================================================
# ENUMS
# ============================================================================


class CompanyStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"


class CompanyType(str, Enum):
    PRIVATE = "private"
    PUBLIC = "public"
    NON_PROFIT = "non_profit"
    GOVERNMENT = "government"
    STARTUP = "startup"
    OTHER = "other"


# ============================================================================
# SCHEMAS
# ============================================================================


class CompanySchemaBase(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


class CompanyCreateRequest(CompanySchemaBase):
    name: str = Field(
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

    company_type: CompanyType = CompanyType.PRIVATE

    country: str = Field(
        default="India",
        min_length=2,
        max_length=100,
    )

    currency: str = Field(
        default="INR",
        min_length=3,
        max_length=3,
    )

    fiscal_year_start_month: int = Field(
        default=4,
        ge=1,
        le=12,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=1000,
    )

    website: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = " ".join(value.split())

        if not value:
            raise ValueError("Company name cannot be empty.")

        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(cls, value: str) -> str:
        return value.upper().strip()


class CompanyUpdateRequest(CompanySchemaBase):
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

    company_type: Optional[CompanyType] = None

    country: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
    )

    currency: Optional[str] = Field(
        default=None,
        min_length=3,
        max_length=3,
    )

    fiscal_year_start_month: Optional[int] = Field(
        default=None,
        ge=1,
        le=12,
    )

    description: Optional[str] = Field(
        default=None,
        max_length=1000,
    )

    website: Optional[str] = Field(
        default=None,
        max_length=500,
    )

    status: Optional[CompanyStatus] = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None

        value = " ".join(value.split())

        if not value:
            raise ValueError("Company name cannot be empty.")

        return value

    @field_validator("currency")
    @classmethod
    def normalize_currency(
        cls,
        value: Optional[str],
    ) -> Optional[str]:
        if value is None:
            return None

        return value.upper().strip()


class CompanyResponse(CompanySchemaBase):
    id: str
    name: str
    legal_name: Optional[str] = None
    industry: Optional[str] = None
    company_type: CompanyType
    country: str
    currency: str
    fiscal_year_start_month: int
    description: Optional[str] = None
    website: Optional[str] = None
    status: CompanyStatus
    owner_id: str
    created_at: datetime
    updated_at: datetime


class CompanyListItem(CompanySchemaBase):
    id: str
    name: str
    industry: Optional[str] = None
    company_type: CompanyType
    country: str
    currency: str
    status: CompanyStatus
    created_at: datetime


class CompanyListResponse(CompanySchemaBase):
    items: list[CompanyListItem]
    total: int
    page: int
    page_size: int


class CompanyStatistics(CompanySchemaBase):
    company_id: str

    total_revenue: Decimal = Decimal("0")
    total_expenses: Decimal = Decimal("0")
    net_profit: Decimal = Decimal("0")
    cash_balance: Decimal = Decimal("0")
    total_assets: Decimal = Decimal("0")
    total_liabilities: Decimal = Decimal("0")
    total_debt: Decimal = Decimal("0")

    active_alerts: int = 0
    critical_alerts: int = 0
    suspicious_transactions: int = 0
    documents_count: int = 0

    last_updated: datetime


class CompanySummary(CompanySchemaBase):
    company: CompanyResponse
    statistics: CompanyStatistics


class CompanyStatusResponse(CompanySchemaBase):
    company_id: str
    status: CompanyStatus
    message: str


# ============================================================================
# INTERNAL MODEL
# ============================================================================


class Company:
    """
    Lightweight company model.

    This is intentionally small so the API can run before the SQLAlchemy
    repository layer is connected.

    Production implementation:
        app/database/models/company.py
        app/database/repositories/company_repository.py
    """

    def __init__(
        self,
        company_id: str,
        name: str,
        owner_id: str,
        legal_name: Optional[str] = None,
        industry: Optional[str] = None,
        company_type: CompanyType = CompanyType.PRIVATE,
        country: str = "India",
        currency: str = "INR",
        fiscal_year_start_month: int = 4,
        description: Optional[str] = None,
        website: Optional[str] = None,
        status_value: CompanyStatus = CompanyStatus.ACTIVE,
    ) -> None:
        now = datetime.now(timezone.utc)

        self.id = company_id
        self.name = name
        self.legal_name = legal_name
        self.industry = industry
        self.company_type = company_type
        self.country = country
        self.currency = currency
        self.fiscal_year_start_month = fiscal_year_start_month
        self.description = description
        self.website = website
        self.status = status_value
        self.owner_id = owner_id
        self.created_at = now
        self.updated_at = now


# ============================================================================
# TEMPORARY STORAGE
# ============================================================================

_COMPANIES: dict[str, Company] = {}


# ============================================================================
# HELPERS
# ============================================================================


def generate_company_id() -> str:
    """
    Generate a company identifier.

    Replace with UUID generation in the database model if preferred.
    """
    import secrets

    return f"cmp_{secrets.token_urlsafe(12)}"


def company_to_response(
    company: Company,
) -> CompanyResponse:
    return CompanyResponse(
        id=company.id,
        name=company.name,
        legal_name=company.legal_name,
        industry=company.industry,
        company_type=company.company_type,
        country=company.country,
        currency=company.currency,
        fiscal_year_start_month=company.fiscal_year_start_month,
        description=company.description,
        website=company.website,
        status=company.status,
        owner_id=company.owner_id,
        created_at=company.created_at,
        updated_at=company.updated_at,
    )


def company_to_list_item(
    company: Company,
) -> CompanyListItem:
    return CompanyListItem(
        id=company.id,
        name=company.name,
        industry=company.industry,
        company_type=company.company_type,
        country=company.country,
        currency=company.currency,
        status=company.status,
        created_at=company.created_at,
    )


def get_company_or_404(
    company_id: str,
) -> Company:
    company = _COMPANIES.get(company_id)

    if company is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Company not found.",
        )

    return company


def user_can_access_company(
    user: AuthUser,
    company: Company,
) -> bool:
    """
    Determine whether a user can access a company.

    Current demo rule:
        - admin can access every company
        - company owner can access the company
        - users whose company_id matches can access it

    Production:
        Use a company_memberships table for many-to-many access.
    """

    if user.role == UserRole.ADMIN:
        return True

    if company.owner_id == user.id:
        return True

    if user.company_id and user.company_id == company.id:
        return True

    return False


def require_company_access(
    user: AuthUser,
    company: Company,
) -> None:
    if not user_can_access_company(user, company):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have access to this company.",
        )


def get_company_statistics(
    company: Company,
) -> CompanyStatistics:
    """
    Return company-level financial statistics.

    This endpoint currently returns a safe placeholder.

    Production implementation should aggregate data from:
        financial statements
        transactions
        forecasts
        fraud alerts
        risk alerts
        documents
    """

    return CompanyStatistics(
        company_id=company.id,
        total_revenue=Decimal("0"),
        total_expenses=Decimal("0"),
        net_profit=Decimal("0"),
        cash_balance=Decimal("0"),
        total_assets=Decimal("0"),
        total_liabilities=Decimal("0"),
        total_debt=Decimal("0"),
        active_alerts=0,
        critical_alerts=0,
        suspicious_transactions=0,
        documents_count=0,
        last_updated=datetime.now(timezone.utc),
    )


# ============================================================================
# CREATE COMPANY
# ============================================================================


@router.post(
    "",
    response_model=CompanyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a company",
)
async def create_company(
    request: CompanyCreateRequest,
    user: AuthUser = Depends(get_current_user),
) -> CompanyResponse:
    """
    Create a new financial workspace/company.

    The authenticated user becomes the company owner.
    """

    # Prevent duplicate company names for the same owner.
    normalized_name = request.name.lower()

    for company in _COMPANIES.values():
        if (
            company.owner_id == user.id
            and company.name.lower() == normalized_name
        ):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="You already have a company with this name.",
            )

    company = Company(
        company_id=generate_company_id(),
        name=request.name,
        legal_name=request.legal_name,
        industry=request.industry,
        company_type=request.company_type,
        country=request.country,
        currency=request.currency,
        fiscal_year_start_month=(
            request.fiscal_year_start_month
        ),
        description=request.description,
        website=request.website,
        owner_id=user.id,
    )

    _COMPANIES[company.id] = company

    return company_to_response(company)


# ============================================================================
# LIST COMPANIES
# ============================================================================


@router.get(
    "",
    response_model=CompanyListResponse,
    summary="List accessible companies",
)
async def list_companies(
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=20,
        ge=1,
        le=100,
    ),
    status_filter: Optional[CompanyStatus] = Query(
        default=None,
        alias="status",
    ),
    user: AuthUser = Depends(get_current_user),
) -> CompanyListResponse:
    """
    List companies accessible by the current user.

    Production implementation should use repository-level filtering
    rather than loading every company into memory.
    """

    accessible: list[Company] = []

    for company in _COMPANIES.values():
        if not user_can_access_company(
            user,
            company,
        ):
            continue

        if (
            status_filter is not None
            and company.status != status_filter
        ):
            continue

        accessible.append(company)

    accessible.sort(
        key=lambda item: item.created_at,
        reverse=True,
    )

    total = len(accessible)

    start = (page - 1) * page_size
    end = start + page_size

    page_items = accessible[start:end]

    return CompanyListResponse(
        items=[
            company_to_list_item(company)
            for company in page_items
        ],
        total=total,
        page=page,
        page_size=page_size,
    )


# ============================================================================
# GET COMPANY
# ============================================================================


@router.get(
    "/{company_id}",
    response_model=CompanyResponse,
    summary="Get company",
)
async def get_company(
    company_id: str,
    user: AuthUser = Depends(get_current_user),
) -> CompanyResponse:
    company = get_company_or_404(company_id)

    require_company_access(
        user,
        company,
    )

    return company_to_response(company)


# ============================================================================
# GET COMPANY SUMMARY
# ============================================================================


@router.get(
    "/{company_id}/summary",
    response_model=CompanySummary,
    summary="Get company financial summary",
)
async def get_company_summary(
    company_id: str,
    user: AuthUser = Depends(get_current_user),
) -> CompanySummary:
    """
    Return company metadata plus high-level financial statistics.

    This becomes the main context endpoint for the dashboard.
    """

    company = get_company_or_404(company_id)

    require_company_access(
        user,
        company,
    )

    statistics = get_company_statistics(
        company
    )

    return CompanySummary(
        company=company_to_response(company),
        statistics=statistics,
    )


# ============================================================================
# UPDATE COMPANY
# ============================================================================


@router.patch(
    "/{company_id}",
    response_model=CompanyResponse,
    summary="Update company",
)
async def update_company(
    company_id: str,
    request: CompanyUpdateRequest,
    user: AuthUser = Depends(get_current_user),
) -> CompanyResponse:
    """
    Update company information.

    Only company owners and administrators may update the company.
    """

    company = get_company_or_404(company_id)

    require_company_access(
        user,
        company,
    )

    if (
        user.role != UserRole.ADMIN
        and company.owner_id != user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the company owner or administrator can update this company.",
        )

    updates = request.model_dump(
        exclude_unset=True
    )

    if "name" in updates:
        new_name = updates["name"].lower()

        for existing in _COMPANIES.values():
            if existing.id == company.id:
                continue

            if (
                existing.owner_id == company.owner_id
                and existing.name.lower() == new_name
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="A company with this name already exists.",
                )

    if "name" in updates:
        company.name = updates["name"]

    if "legal_name" in updates:
        company.legal_name = updates["legal_name"]

    if "industry" in updates:
        company.industry = updates["industry"]

    if "company_type" in updates:
        company.company_type = updates["company_type"]

    if "country" in updates:
        company.country = updates["country"]

    if "currency" in updates:
        company.currency = updates["currency"]

    if "fiscal_year_start_month" in updates:
        company.fiscal_year_start_month = (
            updates["fiscal_year_start_month"]
        )

    if "description" in updates:
        company.description = updates["description"]

    if "website" in updates:
        company.website = updates["website"]

    if "status" in updates:
        company.status = updates["status"]

    company.updated_at = datetime.now(timezone.utc)

    return company_to_response(company)


# ============================================================================
# ACTIVATE COMPANY
# ============================================================================


@router.post(
    "/{company_id}/activate",
    response_model=CompanyStatusResponse,
    summary="Activate company",
)
async def activate_company(
    company_id: str,
    user: AuthUser = Depends(
        require_roles(
            UserRole.ADMIN,
            UserRole.EXECUTIVE,
        )
    ),
) -> CompanyStatusResponse:
    company = get_company_or_404(company_id)

    require_company_access(
        user,
        company,
    )

    company.status = CompanyStatus.ACTIVE
    company.updated_at = datetime.now(timezone.utc)

    return CompanyStatusResponse(
        company_id=company.id,
        status=company.status,
        message="Company activated successfully.",
    )


# ============================================================================
# DEACTIVATE COMPANY
# ============================================================================


@router.post(
    "/{company_id}/deactivate",
    response_model=CompanyStatusResponse,
    summary="Deactivate company",
)
async def deactivate_company(
    company_id: str,
    user: AuthUser = Depends(
        require_roles(
            UserRole.ADMIN,
            UserRole.EXECUTIVE,
        )
    ),
) -> CompanyStatusResponse:
    company = get_company_or_404(company_id)

    require_company_access(
        user,
        company,
    )

    company.status = CompanyStatus.INACTIVE
    company.updated_at = datetime.now(timezone.utc)

    return CompanyStatusResponse(
        company_id=company.id,
        status=company.status,
        message="Company deactivated successfully.",
    )


# ============================================================================
# DELETE COMPANY
# ============================================================================


@router.delete(
    "/{company_id}",
    response_model=CompanyStatusResponse,
    summary="Deactivate a company workspace",
)
async def delete_company(
    company_id: str,
    user: AuthUser = Depends(get_current_user),
) -> CompanyStatusResponse:
    """
    Soft-delete a company.

    Financial systems should generally avoid physical deletion because
    financial records, audit logs, alerts, and documents may need to
    remain traceable.
    """

    company = get_company_or_404(company_id)

    require_company_access(
        user,
        company,
    )

    if (
        user.role != UserRole.ADMIN
        and company.owner_id != user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only the company owner or administrator can deactivate this company.",
        )

    company.status = CompanyStatus.INACTIVE
    company.updated_at = datetime.now(timezone.utc)

    return CompanyStatusResponse(
        company_id=company.id,
        status=company.status,
        message=(
            "Company workspace deactivated. "
            "Financial records were preserved."
        ),
    )


# ============================================================================
# COMPANY STATISTICS
# ============================================================================


@router.get(
    "/{company_id}/statistics",
    response_model=CompanyStatistics,
    summary="Get company statistics",
)
async def company_statistics(
    company_id: str,
    user: AuthUser = Depends(get_current_user),
) -> CompanyStatistics:
    """
    Return aggregated financial and operational statistics.
    """

    company = get_company_or_404(company_id)

    require_company_access(
        user,
        company,
    )

    return get_company_statistics(company)


# ============================================================================
# COMPANY ACCESS CHECK
# ============================================================================


@router.get(
    "/{company_id}/access",
    response_model=dict[str, Any],
    summary="Check company access",
)
async def check_company_access(
    company_id: str,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    Useful for frontend routing and workspace selection.
    """

    company = get_company_or_404(company_id)

    has_access = user_can_access_company(
        user,
        company,
    )

    return {
        "company_id": company.id,
        "user_id": user.id,
        "has_access": has_access,
        "role": user.role.value,
        "company_status": company.status.value,
    }


# ============================================================================
# COMPANY HEALTH
# ============================================================================


@router.get(
    "/{company_id}/health",
    response_model=dict[str, Any],
    summary="Get company workspace health",
)
async def company_health(
    company_id: str,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    """
    High-level health information for the company workspace.

    The actual financial health score should eventually come from the
    financial/risk engine rather than being calculated in this API.
    """

    company = get_company_or_404(company_id)

    require_company_access(
        user,
        company,
    )

    statistics = get_company_statistics(
        company
    )

    return {
        "company_id": company.id,
        "company_name": company.name,
        "status": company.status.value,
        "currency": company.currency,
        "statistics_updated_at": (
            statistics.last_updated
        ),
        "data_available": {
            "financials": (
                statistics.total_revenue != Decimal("0")
                or statistics.total_expenses != Decimal("0")
            ),
            "alerts": statistics.active_alerts > 0,
            "documents": statistics.documents_count > 0,
            "fraud_signals": (
                statistics.suspicious_transactions > 0
            ),
        },
    }


# ============================================================================
# EXPORTS
# ============================================================================


__all__ = [
    "router",
    "CompanyStatus",
    "CompanyType",
    "CompanyCreateRequest",
    "CompanyUpdateRequest",
    "CompanyResponse",
    "CompanyListItem",
    "CompanyListResponse",
    "CompanyStatistics",
    "CompanySummary",
    "CompanyStatusResponse",
    "Company",
]


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("FinCo AI - Company API")
    print("=" * 40)
    print("Router prefix:", router.prefix)
    print()
    print("Endpoints:")
    print("  POST   /companies")
    print("  GET    /companies")
    print("  GET    /companies/{company_id}")
    print("  PATCH  /companies/{company_id}")
    print("  DELETE /companies/{company_id}")
    print("  GET    /companies/{company_id}/summary")
    print("  GET    /companies/{company_id}/statistics")
    print("  GET    /companies/{company_id}/access")
    print("  GET    /companies/{company_id}/health")
    print("  POST   /companies/{company_id}/activate")
    print("  POST   /companies/{company_id}/deactivate")