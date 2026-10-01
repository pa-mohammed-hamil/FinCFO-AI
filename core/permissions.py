"""
FinCo AI - Permission & RBAC System

Responsibilities:
- Role-based access control (RBAC)
- Permission definitions
- Company/tenant isolation
- Resource-level authorization
- Agent/tool authorization
- Financial operation protection
- Human-review requirements

Important:
    Permissions are NOT a replacement for authentication.
    Authentication identifies the user.
    This module decides what the authenticated user can do.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional

from app.core.exceptions import (
    AuthorizationError,
    PermissionDeniedError,
    TenantAccessError,
)


# ============================================================
# Roles
# ============================================================

class Role(str, Enum):
    """
    FinCo AI application roles.

    Higher-level roles are not automatically allowed to
    perform every sensitive operation; explicit permissions
    remain the source of truth.
    """

    SUPER_ADMIN = "super_admin"

    COMPANY_ADMIN = "company_admin"

    FINANCE_MANAGER = "finance_manager"

    FINANCE_ANALYST = "finance_analyst"

    RISK_MANAGER = "risk_manager"

    FRAUD_ANALYST = "fraud_analyst"

    EXECUTIVE = "executive"

    AUDITOR = "auditor"

    VIEWER = "viewer"


# ============================================================
# Permissions
# ============================================================

class Permission(str, Enum):
    # --------------------------------------------------------
    # Company
    # --------------------------------------------------------

    COMPANY_READ = "company:read"
    COMPANY_UPDATE = "company:update"
    COMPANY_DELETE = "company:delete"

    # --------------------------------------------------------
    # Users
    # --------------------------------------------------------

    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"

    # --------------------------------------------------------
    # Documents / RAG
    # --------------------------------------------------------

    DOCUMENT_READ = "document:read"
    DOCUMENT_UPLOAD = "document:upload"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_PROCESS = "document:process"

    RAG_QUERY = "rag:query"

    # --------------------------------------------------------
    # Financial Data
    # --------------------------------------------------------

    FINANCIAL_READ = "financial:read"
    FINANCIAL_CREATE = "financial:create"
    FINANCIAL_UPDATE = "financial:update"
    FINANCIAL_DELETE = "financial:delete"

    REVENUE_READ = "revenue:read"
    EXPENSE_READ = "expense:read"
    PNL_READ = "pnl:read"
    BALANCE_SHEET_READ = "balance_sheet:read"
    CASH_FLOW_READ = "cash_flow:read"

    # --------------------------------------------------------
    # Transactions
    # --------------------------------------------------------

    TRANSACTION_READ = "transaction:read"
    TRANSACTION_CREATE = "transaction:create"
    TRANSACTION_UPDATE = "transaction:update"
    TRANSACTION_DELETE = "transaction:delete"

    # --------------------------------------------------------
    # Forecasting
    # --------------------------------------------------------

    FORECAST_READ = "forecast:read"
    FORECAST_RUN = "forecast:run"

    # --------------------------------------------------------
    # Fraud
    # --------------------------------------------------------

    FRAUD_READ = "fraud:read"
    FRAUD_INVESTIGATE = "fraud:investigate"
    FRAUD_REVIEW = "fraud:review"

    # --------------------------------------------------------
    # Risk
    # --------------------------------------------------------

    RISK_READ = "risk:read"
    RISK_ANALYZE = "risk:analyze"

    # --------------------------------------------------------
    # Alerts
    # --------------------------------------------------------

    ALERT_READ = "alert:read"
    ALERT_CREATE = "alert:create"
    ALERT_UPDATE = "alert:update"
    ALERT_REVIEW = "alert:review"
    ALERT_RESOLVE = "alert:resolve"

    # --------------------------------------------------------
    # Recommendations
    # --------------------------------------------------------

    RECOMMENDATION_READ = "recommendation:read"
    RECOMMENDATION_CREATE = "recommendation:create"
    RECOMMENDATION_APPROVE = "recommendation:approve"
    RECOMMENDATION_REJECT = "recommendation:reject"

    # --------------------------------------------------------
    # What-if / Simulation
    # --------------------------------------------------------

    WHAT_IF_READ = "what_if:read"
    WHAT_IF_RUN = "what_if:run"

    # --------------------------------------------------------
    # Copilot / Agents
    # --------------------------------------------------------

    COPILOT_QUERY = "copilot:query"
    AGENT_EXECUTE = "agent:execute"
    AGENT_FINANCIAL = "agent:financial"
    AGENT_FRAUD = "agent:fraud"
    AGENT_FORECAST = "agent:forecast"
    AGENT_RAG = "agent:rag"
    AGENT_WHAT_IF = "agent:what_if"
    AGENT_RECOMMENDATION = "agent:recommendation"

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    REPORT_READ = "report:read"
    REPORT_GENERATE = "report:generate"
    REPORT_EXPORT = "report:export"

    # --------------------------------------------------------
    # Audit / Security
    # --------------------------------------------------------

    AUDIT_READ = "audit:read"
    AUDIT_EXPORT = "audit:export"

    SECURITY_READ = "security:read"
    SECURITY_MANAGE = "security:manage"

    # --------------------------------------------------------
    # Sensitive Operations
    # --------------------------------------------------------

    FINANCIAL_APPROVE = "financial:approve"
    PAYMENT_APPROVE = "payment:approve"
    BANK_ACCOUNT_CHANGE = "bank_account:change"

    # --------------------------------------------------------
    # Administration
    # --------------------------------------------------------

    SYSTEM_ADMIN = "system:admin"


# ============================================================
# Permission Groups
# ============================================================

READ_ONLY_PERMISSIONS = {
    Permission.COMPANY_READ,
    Permission.DOCUMENT_READ,
    Permission.RAG_QUERY,

    Permission.FINANCIAL_READ,
    Permission.REVENUE_READ,
    Permission.EXPENSE_READ,
    Permission.PNL_READ,
    Permission.BALANCE_SHEET_READ,
    Permission.CASH_FLOW_READ,

    Permission.TRANSACTION_READ,

    Permission.FORECAST_READ,

    Permission.FRAUD_READ,
    Permission.RISK_READ,

    Permission.ALERT_READ,

    Permission.RECOMMENDATION_READ,

    Permission.WHAT_IF_READ,

    Permission.REPORT_READ,

    Permission.COPILOT_QUERY,
}


FINANCE_PERMISSIONS = {
    Permission.COMPANY_READ,

    Permission.FINANCIAL_READ,
    Permission.FINANCIAL_CREATE,
    Permission.FINANCIAL_UPDATE,

    Permission.REVENUE_READ,
    Permission.EXPENSE_READ,
    Permission.PNL_READ,
    Permission.BALANCE_SHEET_READ,
    Permission.CASH_FLOW_READ,

    Permission.TRANSACTION_READ,
    Permission.TRANSACTION_CREATE,
    Permission.TRANSACTION_UPDATE,

    Permission.FORECAST_READ,
    Permission.FORECAST_RUN,

    Permission.WHAT_IF_READ,
    Permission.WHAT_IF_RUN,

    Permission.RECOMMENDATION_READ,
    Permission.RECOMMENDATION_CREATE,

    Permission.REPORT_READ,
    Permission.REPORT_GENERATE,

    Permission.COPILOT_QUERY,
    Permission.AGENT_EXECUTE,
    Permission.AGENT_FINANCIAL,
    Permission.AGENT_FORECAST,
    Permission.AGENT_RAG,
    Permission.AGENT_WHAT_IF,
}


# ============================================================
# Role → Permission Mapping
# ============================================================

ROLE_PERMISSIONS: dict[Role, set[Permission]] = {

    # --------------------------------------------------------
    # Super Admin
    # --------------------------------------------------------

    Role.SUPER_ADMIN: set(Permission),

    # --------------------------------------------------------
    # Company Admin
    # --------------------------------------------------------

    Role.COMPANY_ADMIN: {
        Permission.COMPANY_READ,
        Permission.COMPANY_UPDATE,

        Permission.USER_READ,
        Permission.USER_CREATE,
        Permission.USER_UPDATE,
        Permission.USER_DELETE,

        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_UPLOAD,
        Permission.DOCUMENT_DELETE,
        Permission.DOCUMENT_PROCESS,

        Permission.RAG_QUERY,

        Permission.FINANCIAL_READ,
        Permission.FINANCIAL_CREATE,
        Permission.FINANCIAL_UPDATE,

        Permission.TRANSACTION_READ,
        Permission.TRANSACTION_CREATE,
        Permission.TRANSACTION_UPDATE,
        Permission.TRANSACTION_DELETE,

        Permission.FORECAST_READ,
        Permission.FORECAST_RUN,

        Permission.FRAUD_READ,
        Permission.FRAUD_INVESTIGATE,
        Permission.FRAUD_REVIEW,

        Permission.RISK_READ,
        Permission.RISK_ANALYZE,

        Permission.ALERT_READ,
        Permission.ALERT_CREATE,
        Permission.ALERT_UPDATE,
        Permission.ALERT_REVIEW,
        Permission.ALERT_RESOLVE,

        Permission.RECOMMENDATION_READ,
        Permission.RECOMMENDATION_CREATE,
        Permission.RECOMMENDATION_APPROVE,
        Permission.RECOMMENDATION_REJECT,

        Permission.WHAT_IF_READ,
        Permission.WHAT_IF_RUN,

        Permission.COPILOT_QUERY,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_FINANCIAL,
        Permission.AGENT_FRAUD,
        Permission.AGENT_FORECAST,
        Permission.AGENT_RAG,
        Permission.AGENT_WHAT_IF,
        Permission.AGENT_RECOMMENDATION,

        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,
        Permission.REPORT_EXPORT,

        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,

        Permission.SECURITY_READ,
    },

    # --------------------------------------------------------
    # Finance Manager
    # --------------------------------------------------------

    Role.FINANCE_MANAGER: (
        FINANCE_PERMISSIONS
        | {
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_UPLOAD,
            Permission.DOCUMENT_PROCESS,

            Permission.FRAUD_READ,
            Permission.RISK_READ,

            Permission.ALERT_READ,
            Permission.ALERT_REVIEW,
            Permission.ALERT_RESOLVE,

            Permission.RECOMMENDATION_APPROVE,
            Permission.RECOMMENDATION_REJECT,

            Permission.REPORT_EXPORT,
        }
    ),

    # --------------------------------------------------------
    # Finance Analyst
    # --------------------------------------------------------

    Role.FINANCE_ANALYST: (
        FINANCE_PERMISSIONS
        | {
            Permission.DOCUMENT_READ,
            Permission.DOCUMENT_UPLOAD,

            Permission.FRAUD_READ,
            Permission.RISK_READ,

            Permission.ALERT_READ,

            Permission.REPORT_EXPORT,
        }
    ),

    # --------------------------------------------------------
    # Risk Manager
    # --------------------------------------------------------

    Role.RISK_MANAGER: {
        Permission.COMPANY_READ,

        Permission.FINANCIAL_READ,
        Permission.REVENUE_READ,
        Permission.EXPENSE_READ,
        Permission.PNL_READ,
        Permission.BALANCE_SHEET_READ,
        Permission.CASH_FLOW_READ,

        Permission.TRANSACTION_READ,

        Permission.FORECAST_READ,
        Permission.FORECAST_RUN,

        Permission.FRAUD_READ,
        Permission.FRAUD_INVESTIGATE,
        Permission.FRAUD_REVIEW,

        Permission.RISK_READ,
        Permission.RISK_ANALYZE,

        Permission.ALERT_READ,
        Permission.ALERT_CREATE,
        Permission.ALERT_UPDATE,
        Permission.ALERT_REVIEW,
        Permission.ALERT_RESOLVE,

        Permission.RECOMMENDATION_READ,
        Permission.RECOMMENDATION_CREATE,

        Permission.WHAT_IF_READ,
        Permission.WHAT_IF_RUN,

        Permission.COPILOT_QUERY,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_FRAUD,
        Permission.AGENT_FORECAST,
        Permission.AGENT_RAG,
        Permission.AGENT_WHAT_IF,
        Permission.AGENT_RECOMMENDATION,

        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,
        Permission.REPORT_EXPORT,

        Permission.AUDIT_READ,
    },

    # --------------------------------------------------------
    # Fraud Analyst
    # --------------------------------------------------------

    Role.FRAUD_ANALYST: {
        Permission.COMPANY_READ,

        Permission.FINANCIAL_READ,
        Permission.TRANSACTION_READ,

        Permission.FRAUD_READ,
        Permission.FRAUD_INVESTIGATE,
        Permission.FRAUD_REVIEW,

        Permission.RISK_READ,
        Permission.RISK_ANALYZE,

        Permission.ALERT_READ,
        Permission.ALERT_REVIEW,
        Permission.ALERT_RESOLVE,

        Permission.RECOMMENDATION_READ,

        Permission.COPILOT_QUERY,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_FRAUD,
        Permission.AGENT_RAG,

        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,

        Permission.AUDIT_READ,
    },

    # --------------------------------------------------------
    # Executive
    # --------------------------------------------------------

    Role.EXECUTIVE: {
        Permission.COMPANY_READ,

        Permission.FINANCIAL_READ,
        Permission.REVENUE_READ,
        Permission.EXPENSE_READ,
        Permission.PNL_READ,
        Permission.BALANCE_SHEET_READ,
        Permission.CASH_FLOW_READ,

        Permission.TRANSACTION_READ,

        Permission.FORECAST_READ,
        Permission.FRAUD_READ,
        Permission.RISK_READ,

        Permission.ALERT_READ,

        Permission.RECOMMENDATION_READ,
        Permission.RECOMMENDATION_APPROVE,
        Permission.RECOMMENDATION_REJECT,

        Permission.WHAT_IF_READ,
        Permission.WHAT_IF_RUN,

        Permission.COPILOT_QUERY,
        Permission.AGENT_EXECUTE,
        Permission.AGENT_FINANCIAL,
        Permission.AGENT_FORECAST,
        Permission.AGENT_RAG,
        Permission.AGENT_WHAT_IF,
        Permission.AGENT_RECOMMENDATION,

        Permission.REPORT_READ,
        Permission.REPORT_GENERATE,
        Permission.REPORT_EXPORT,
    },

    # --------------------------------------------------------
    # Auditor
    # --------------------------------------------------------

    Role.AUDITOR: {
        Permission.COMPANY_READ,

        Permission.FINANCIAL_READ,
        Permission.REVENUE_READ,
        Permission.EXPENSE_READ,
        Permission.PNL_READ,
        Permission.BALANCE_SHEET_READ,
        Permission.CASH_FLOW_READ,

        Permission.TRANSACTION_READ,

        Permission.DOCUMENT_READ,

        Permission.FORECAST_READ,

        Permission.FRAUD_READ,
        Permission.RISK_READ,

        Permission.ALERT_READ,

        Permission.RECOMMENDATION_READ,

        Permission.REPORT_READ,
        Permission.REPORT_EXPORT,

        Permission.AUDIT_READ,
        Permission.AUDIT_EXPORT,

        Permission.SECURITY_READ,
    },

    # --------------------------------------------------------
    # Viewer
    # --------------------------------------------------------

    Role.VIEWER: READ_ONLY_PERMISSIONS,
}


# ============================================================
# Protected / High-Risk Permissions
# ============================================================

SENSITIVE_PERMISSIONS = {
    Permission.PAYMENT_APPROVE,
    Permission.BANK_ACCOUNT_CHANGE,
    Permission.FINANCIAL_APPROVE,
    Permission.FINANCIAL_DELETE,
    Permission.TRANSACTION_DELETE,
    Permission.USER_DELETE,
    Permission.COMPANY_DELETE,
    Permission.SECURITY_MANAGE,
    Permission.SYSTEM_ADMIN,
}


HUMAN_REVIEW_PERMISSIONS = {
    Permission.PAYMENT_APPROVE,
    Permission.BANK_ACCOUNT_CHANGE,
    Permission.FINANCIAL_APPROVE,
}


# ============================================================
# User Authorization Context
# ============================================================

@dataclass
class AuthorizationContext:
    """
    Information required to make an authorization decision.
    """

    user_id: str

    company_id: Optional[str]

    roles: set[Role]

    is_active: bool = True

    is_superuser: bool = False

    metadata: Optional[dict[str, Any]] = None

    @property
    def permissions(self) -> set[Permission]:
        """Return all permissions granted by the user's roles."""

        if self.is_superuser:
            return set(Permission)

        permissions: set[Permission] = set()

        for role in self.roles:
            permissions.update(
                ROLE_PERMISSIONS.get(
                    role,
                    set(),
                )
            )

        return permissions


# ============================================================
# Permission Service
# ============================================================

class PermissionService:
    """
    Central authorization service for FinCo AI.
    """

    @staticmethod
    def has_role(
        context: AuthorizationContext,
        role: Role,
    ) -> bool:
        """Check whether a user has a specific role."""

        if context.is_superuser:
            return True

        return role in context.roles

    @staticmethod
    def has_permission(
        context: AuthorizationContext,
        permission: Permission,
    ) -> bool:
        """Check whether a user has a specific permission."""

        if not context.is_active:
            return False

        if context.is_superuser:
            return True

        return permission in context.permissions

    @staticmethod
    def has_any_permission(
        context: AuthorizationContext,
        permissions: Iterable[Permission],
    ) -> bool:
        """Return True when the user has at least one permission."""

        return any(
            PermissionService.has_permission(
                context,
                permission,
            )
            for permission in permissions
        )

    @staticmethod
    def has_all_permissions(
        context: AuthorizationContext,
        permissions: Iterable[Permission],
    ) -> bool:
        """Return True only when all permissions are granted."""

        return all(
            PermissionService.has_permission(
                context,
                permission,
            )
            for permission in permissions
        )

    @staticmethod
    def require_permission(
        context: AuthorizationContext,
        permission: Permission,
    ) -> None:
        """
        Raise an exception if permission is missing.
        """

        if not context.is_active:
            raise AuthorizationError(
                "User account is inactive."
            )

        if not PermissionService.has_permission(
            context,
            permission,
        ):
            raise PermissionDeniedError(
                message=(
                    f"Permission denied: "
                    f"{permission.value}"
                ),
                details={
                    "user_id": context.user_id,
                    "permission": permission.value,
                },
            )

    @staticmethod
    def require_any_permission(
        context: AuthorizationContext,
        permissions: Iterable[Permission],
    ) -> None:
        """Require at least one of the supplied permissions."""

        permissions = list(permissions)

        if not PermissionService.has_any_permission(
            context,
            permissions,
        ):
            raise PermissionDeniedError(
                message="None of the required permissions were granted.",
                details={
                    "required_permissions": [
                        permission.value
                        for permission in permissions
                    ]
                },
            )

    @staticmethod
    def require_role(
        context: AuthorizationContext,
        role: Role,
    ) -> None:
        """Require a specific role."""

        if not PermissionService.has_role(
            context,
            role,
        ):
            raise PermissionDeniedError(
                message=(
                    f"Role required: {role.value}"
                ),
                details={
                    "user_id": context.user_id,
                    "required_role": role.value,
                },
            )


# ============================================================
# Tenant / Company Isolation
# ============================================================

class TenantAccessService:
    """
    Enforces company-level multi-tenant isolation.

    Every company-scoped resource should pass through this
    check before being read or modified.
    """

    @staticmethod
    def can_access_company(
        context: AuthorizationContext,
        company_id: str,
    ) -> bool:
        """
        Determine whether the user can access a company.
        """

        if not context.is_active:
            return False

        if context.is_superuser:
            return True

        if not context.company_id:
            return False

        return str(
            context.company_id
        ) == str(company_id)

    @staticmethod
    def require_company_access(
        context: AuthorizationContext,
        company_id: str,
    ) -> None:
        """
        Raise TenantAccessError when company access is denied.
        """

        if not TenantAccessService.can_access_company(
            context,
            company_id,
        ):
            raise TenantAccessError(
                message="User does not have access to this company.",
                details={
                    "user_id": context.user_id,
                    "requested_company_id": str(company_id),
                    "user_company_id": (
                        str(context.company_id)
                        if context.company_id
                        else None
                    ),
                },
            )

    @staticmethod
    def require_company_and_permission(
        context: AuthorizationContext,
        company_id: str,
        permission: Permission,
    ) -> None:
        """
        Combined tenant + permission check.
        """

        TenantAccessService.require_company_access(
            context,
            company_id,
        )

        PermissionService.require_permission(
            context,
            permission,
        )


# ============================================================
# Resource Authorization
# ============================================================

class ResourcePermissionService:
    """
    Resource-level authorization helpers.
    """

    @staticmethod
    def can_read_financial_data(
        context: AuthorizationContext,
        company_id: str,
    ) -> bool:

        return (
            TenantAccessService.can_access_company(
                context,
                company_id,
            )
            and PermissionService.has_permission(
                context,
                Permission.FINANCIAL_READ,
            )
        )

    @staticmethod
    def can_modify_financial_data(
        context: AuthorizationContext,
        company_id: str,
    ) -> bool:

        return (
            TenantAccessService.can_access_company(
                context,
                company_id,
            )
            and PermissionService.has_permission(
                context,
                Permission.FINANCIAL_UPDATE,
            )
        )

    @staticmethod
    def can_read_transaction(
        context: AuthorizationContext,
        company_id: str,
    ) -> bool:

        return (
            TenantAccessService.can_access_company(
                context,
                company_id,
            )
            and PermissionService.has_permission(
                context,
                Permission.TRANSACTION_READ,
            )
        )

    @staticmethod
    def can_investigate_fraud(
        context: AuthorizationContext,
        company_id: str,
    ) -> bool:

        return (
            TenantAccessService.can_access_company(
                context,
                company_id,
            )
            and PermissionService.has_permission(
                context,
                Permission.FRAUD_INVESTIGATE,
            )
        )

    @staticmethod
    def can_run_forecast(
        context: AuthorizationContext,
        company_id: str,
    ) -> bool:

        return (
            TenantAccessService.can_access_company(
                context,
                company_id,
            )
            and PermissionService.has_permission(
                context,
                Permission.FORECAST_RUN,
            )
        )

    @staticmethod
    def can_run_what_if(
        context: AuthorizationContext,
        company_id: str,
    ) -> bool:

        return (
            TenantAccessService.can_access_company(
                context,
                company_id,
            )
            and PermissionService.has_permission(
                context,
                Permission.WHAT_IF_RUN,
            )
        )


# ============================================================
# Agent Authorization
# ============================================================

class AgentPermissionService:
    """
    Controls which specialized agents a user can execute.
    """

    AGENT_PERMISSIONS = {
        "financial": Permission.AGENT_FINANCIAL,
        "fraud": Permission.AGENT_FRAUD,
        "forecast": Permission.AGENT_FORECAST,
        "rag": Permission.AGENT_RAG,
        "what_if": Permission.AGENT_WHAT_IF,
        "recommendation": Permission.AGENT_RECOMMENDATION,
    }

    @classmethod
    def can_execute_agent(
        cls,
        context: AuthorizationContext,
        agent_name: str,
    ) -> bool:

        if not PermissionService.has_permission(
            context,
            Permission.AGENT_EXECUTE,
        ):
            return False

        permission = cls.AGENT_PERMISSIONS.get(
            agent_name.lower()
        )

        if permission is None:
            return False

        return PermissionService.has_permission(
            context,
            permission,
        )

    @classmethod
    def require_agent_access(
        cls,
        context: AuthorizationContext,
        agent_name: str,
    ) -> None:

        if not cls.can_execute_agent(
            context,
            agent_name,
        ):
            raise PermissionDeniedError(
                message=(
                    f"User is not authorized to execute "
                    f"agent: {agent_name}"
                ),
                details={
                    "agent": agent_name,
                    "user_id": context.user_id,
                },
            )


# ============================================================
# Tool Authorization
# ============================================================

class ToolPermissionService:
    """
    Protects agent tools.

    This is important because an agent must never gain more
    privileges merely because an LLM decided to call a tool.
    """

    TOOL_PERMISSIONS = {
        "get_financial_metrics": Permission.FINANCIAL_READ,
        "get_pnl": Permission.PNL_READ,
        "get_balance_sheet": Permission.BALANCE_SHEET_READ,
        "get_cash_flow": Permission.CASH_FLOW_READ,

        "search_transactions": Permission.TRANSACTION_READ,

        "run_forecast": Permission.FORECAST_RUN,

        "detect_fraud": Permission.FRAUD_INVESTIGATE,

        "search_documents": Permission.RAG_QUERY,

        "run_what_if": Permission.WHAT_IF_RUN,

        "create_recommendation": (
            Permission.RECOMMENDATION_CREATE
        ),

        "approve_recommendation": (
            Permission.RECOMMENDATION_APPROVE
        ),

        "generate_report": Permission.REPORT_GENERATE,

        "export_report": Permission.REPORT_EXPORT,
    }

    @classmethod
    def can_execute_tool(
        cls,
        context: AuthorizationContext,
        tool_name: str,
    ) -> bool:

        permission = cls.TOOL_PERMISSIONS.get(
            tool_name
        )

        if permission is None:
            return False

        return PermissionService.has_permission(
            context,
            permission,
        )

    @classmethod
    def require_tool_access(
        cls,
        context: AuthorizationContext,
        tool_name: str,
    ) -> None:

        if not cls.can_execute_tool(
            context,
            tool_name,
        ):
            raise PermissionDeniedError(
                message=(
                    f"User is not authorized to execute "
                    f"tool: {tool_name}"
                ),
                details={
                    "tool": tool_name,
                    "user_id": context.user_id,
                },
            )


# ============================================================
# Sensitive Operation Protection
# ============================================================

class SensitiveOperationService:
    """
    Additional protection for high-impact operations.

    These operations should generally require human review,
    even when the user has the underlying permission.
    """

    @staticmethod
    def requires_human_review(
        permission: Permission,
    ) -> bool:

        return permission in HUMAN_REVIEW_PERMISSIONS

    @staticmethod
    def require_sensitive_access(
        context: AuthorizationContext,
        permission: Permission,
        human_review_approved: bool = False,
    ) -> None:

        PermissionService.require_permission(
            context,
            permission,
        )

        if (
            permission in HUMAN_REVIEW_PERMISSIONS
            and not human_review_approved
        ):
            raise AuthorizationError(
                message=(
                    "This operation requires human approval."
                ),
                details={
                    "permission": permission.value,
                    "human_review_required": True,
                },
            )


# ============================================================
# Convenience Functions
# ============================================================

def require_permission(
    context: AuthorizationContext,
    permission: Permission,
) -> None:
    """
    Shortcut for PermissionService.require_permission().
    """

    PermissionService.require_permission(
        context,
        permission,
    )


def require_company_access(
    context: AuthorizationContext,
    company_id: str,
) -> None:
    """
    Shortcut for tenant validation.
    """

    TenantAccessService.require_company_access(
        context,
        company_id,
    )


def require_company_permission(
    context: AuthorizationContext,
    company_id: str,
    permission: Permission,
) -> None:
    """
    Shortcut for combined company + permission validation.
    """

    TenantAccessService.require_company_and_permission(
        context,
        company_id,
        permission,
    )


def get_user_permissions(
    context: AuthorizationContext,
) -> list[str]:
    """
    Return permissions for API/UI display.
    """

    return sorted(
        permission.value
        for permission in context.permissions
    )


def get_user_roles(
    context: AuthorizationContext,
) -> list[str]:
    """
    Return role names for API/UI display.
    """

    return sorted(
        role.value
        for role in context.roles
    )


# ============================================================
# Permission Matrix
# ============================================================

def get_role_permission_matrix() -> dict[str, list[str]]:
    """
    Return the complete RBAC matrix.

    Useful for:
    - Admin UI
    - documentation
    - testing
    - debugging
    """

    return {
        role.value: sorted(
            permission.value
            for permission in permissions
        )
        for role, permissions in ROLE_PERMISSIONS.items()
    }