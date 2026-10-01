"""
FinCo AI - Database Agent Tools

Path:
    backend/app/agents/tools/database_tools.py

Purpose:
    Provides safe, read-oriented database tools for FinCo AI agents.

Architecture:

    Agent
      |
      v
    Tool Registry
      |
      v
    database_tools.py
      |
      +--> Company Repository
      +--> Financial Repository
      +--> Transaction Repository
      +--> Document Repository
      +--> Customer Repository
      +--> Alert Repository
      |
      v
    Database / ORM

Important:
    Agents must NOT receive raw database sessions or execute
    arbitrary SQL.

Responsibilities:
    - Retrieve company information.
    - Retrieve financial records.
    - Retrieve transactions.
    - Retrieve documents.
    - Retrieve customers.
    - Retrieve alerts.
    - Search repository data.
    - Apply company/user scope.
    - Validate limits and identifiers.
    - Return JSON-friendly results.

This module intentionally avoids:
    - Raw SQL.
    - LLM-generated SQL.
    - Direct ORM queries.
    - Database writes.
    - Financial calculations.
    - Fraud scoring.
    - RAG retrieval logic.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


# ============================================================================
# Exceptions
# ============================================================================


class DatabaseToolError(Exception):
    """Base exception for database tools."""


class InvalidDatabaseToolInputError(
    DatabaseToolError
):
    """Raised when database-tool input is invalid."""


class DatabaseToolExecutionError(
    DatabaseToolError
):
    """Raised when a repository operation fails."""


class DatabaseToolConfigurationError(
    DatabaseToolError
):
    """Raised when required repositories are unavailable."""


class DatabaseAccessDeniedError(
    DatabaseToolError
):
    """Raised when database access is not authorized."""


# ============================================================================
# Constants
# ============================================================================


DEFAULT_LIMIT = 50

MAX_LIMIT = 500

DEFAULT_OFFSET = 0

SUPPORTED_ENTITY_TYPES = {
    "company",
    "financial",
    "transaction",
    "document",
    "customer",
    "alert",
}


# ============================================================================
# Result Model
# ============================================================================


class DatabaseToolResult:
    """
    Standard response envelope for database tools.

    A simple class is used instead of requiring Pydantic so this
    tool module remains independent from the API/schema layer.
    """

    def __init__(
        self,
        *,
        success: bool,
        operation: str,
        message: str,
        data: Any = None,
        error: Optional[str] = None,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> None:

        self.success = bool(success)

        self.operation = operation

        self.message = message

        self.data = data

        self.error = error

        self.timestamp = (
            datetime.now(
                timezone.utc
            )
        )

        self.metadata = dict(
            metadata or {}
        )

    def to_dict(self) -> dict[str, Any]:

        return {
            "success":
                self.success,

            "operation":
                self.operation,

            "message":
                self.message,

            "data":
                _serialize(self.data),

            "error":
                self.error,

            "timestamp":
                self.timestamp.isoformat(),

            "metadata":
                _serialize(
                    self.metadata
                ),
        }


# ============================================================================
# Database Tools
# ============================================================================


class DatabaseTools:
    """
    Agent-facing database access toolkit.

    Repositories are injected into this class.

    Example:

        tools = DatabaseTools(
            company_repository=company_repository,
            financial_repository=financial_repository,
            transaction_repository=transaction_repository,
            document_repository=document_repository,
            customer_repository=customer_repository,
            alert_repository=alert_repository,
        )

    Agents interact with these tools instead of the repositories
    directly.
    """

    def __init__(
        self,
        *,
        company_repository: Any = None,
        financial_repository: Any = None,
        transaction_repository: Any = None,
        document_repository: Any = None,
        customer_repository: Any = None,
        alert_repository: Any = None,
        authorization_service: Any = None,
        audit_service: Any = None,
    ) -> None:

        self.company_repository = (
            company_repository
        )

        self.financial_repository = (
            financial_repository
        )

        self.transaction_repository = (
            transaction_repository
        )

        self.document_repository = (
            document_repository
        )

        self.customer_repository = (
            customer_repository
        )

        self.alert_repository = (
            alert_repository
        )

        self.authorization_service = (
            authorization_service
        )

        self.audit_service = (
            audit_service
        )

    # ========================================================================
    # COMPANY
    # ========================================================================

    def get_company(
        self,
        *,
        company_id: str,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve a company by ID."""

        self._validate_company_id(
            company_id
        )

        self._authorize(
            user_id=user_id,
            action="read_company",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.company_repository,
            "company_repository",
        )

        try:

            result = self._repository_call(
                repository,
                (
                    "get_by_id",
                    "find_by_id",
                    "get_company",
                    "find",
                ),
                company_id,
            )

            return self._success(
                operation="get_company",
                message=(
                    "Company retrieved successfully."
                ),
                data=result,
                metadata={
                    "company_id":
                        company_id,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve company: {exc}"
            ) from exc

    def list_companies(
        self,
        *,
        user_id: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = DEFAULT_OFFSET,
    ) -> dict[str, Any]:
        """List companies accessible to the current user."""

        limit = self._validate_limit(
            limit
        )

        offset = self._validate_offset(
            offset
        )

        self._authorize(
            user_id=user_id,
            action="list_companies",
        )

        repository = self._require_repository(
            self.company_repository,
            "company_repository",
        )

        try:

            result = self._repository_list(
                repository,
                (
                    "list",
                    "get_all",
                    "list_companies",
                ),
                limit=limit,
                offset=offset,
            )

            return self._success(
                operation="list_companies",
                message=(
                    "Companies retrieved successfully."
                ),
                data=result,
                metadata={
                    "limit":
                        limit,

                    "offset":
                        offset,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to list companies: {exc}"
            ) from exc

    # ========================================================================
    # FINANCIAL DATA
    # ========================================================================

    def get_financial_records(
        self,
        *,
        company_id: str,
        start_date: Optional[
            str
        ] = None,
        end_date: Optional[
            str
        ] = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = DEFAULT_OFFSET,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Retrieve financial records for a company.

        Date filtering is passed to the repository when supported.
        """

        self._validate_company_id(
            company_id
        )

        limit = self._validate_limit(
            limit
        )

        offset = self._validate_offset(
            offset
        )

        start = self._validate_date(
            start_date,
            "start_date",
        )

        end = self._validate_date(
            end_date,
            "end_date",
        )

        self._validate_date_range(
            start,
            end,
        )

        self._authorize(
            user_id=user_id,
            action="read_financial_data",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.financial_repository,
            "financial_repository",
        )

        payload = {
            "company_id":
                company_id,

            "start_date":
                start,

            "end_date":
                end,

            "limit":
                limit,

            "offset":
                offset,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_company_financials",
                    "get_financial_records",
                    "list_by_company",
                    "find_by_company",
                    "list",
                ),
                payload,
            )

            return self._success(
                operation="get_financial_records",
                message=(
                    "Financial records retrieved."
                ),
                data=result,
                metadata={
                    "company_id":
                        company_id,

                    "limit":
                        limit,

                    "offset":
                        offset,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve financial records: {exc}"
            ) from exc

    def get_latest_financials(
        self,
        *,
        company_id: str,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve the latest available financial record."""

        self._validate_company_id(
            company_id
        )

        self._authorize(
            user_id=user_id,
            action="read_financial_data",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.financial_repository,
            "financial_repository",
        )

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_latest",
                    "get_latest_financials",
                    "latest_by_company",
                    "find_latest",
                ),
                {
                    "company_id":
                        company_id,
                },
            )

            return self._success(
                operation="get_latest_financials",
                message=(
                    "Latest financial record retrieved."
                ),
                data=result,
                metadata={
                    "company_id":
                        company_id,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve latest financials: {exc}"
            ) from exc

    # ========================================================================
    # TRANSACTIONS
    # ========================================================================

    def get_transactions(
        self,
        *,
        company_id: str,
        start_date: Optional[
            str
        ] = None,
        end_date: Optional[
            str
        ] = None,
        transaction_type: Optional[
            str
        ] = None,
        status: Optional[
            str
        ] = None,
        min_amount: Optional[
            float
        ] = None,
        max_amount: Optional[
            float
        ] = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = DEFAULT_OFFSET,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Retrieve company transactions.

        This is intentionally a parameterized repository operation.
        The agent cannot supply arbitrary SQL.
        """

        self._validate_company_id(
            company_id
        )

        limit = self._validate_limit(
            limit
        )

        offset = self._validate_offset(
            offset
        )

        start = self._validate_date(
            start_date,
            "start_date",
        )

        end = self._validate_date(
            end_date,
            "end_date",
        )

        self._validate_date_range(
            start,
            end,
        )

        minimum = self._optional_number(
            min_amount,
            "min_amount",
        )

        maximum = self._optional_number(
            max_amount,
            "max_amount",
        )

        if (
            minimum is not None
            and maximum is not None
            and minimum > maximum
        ):

            raise InvalidDatabaseToolInputError(
                "min_amount cannot exceed max_amount."
            )

        self._authorize(
            user_id=user_id,
            action="read_transactions",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.transaction_repository,
            "transaction_repository",
        )

        payload = {
            "company_id":
                company_id,

            "start_date":
                start,

            "end_date":
                end,

            "transaction_type":
                transaction_type,

            "status":
                status,

            "min_amount":
                minimum,

            "max_amount":
                maximum,

            "limit":
                limit,

            "offset":
                offset,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_company_transactions",
                    "get_transactions",
                    "list_by_company",
                    "find_by_company",
                    "list",
                ),
                payload,
            )

            return self._success(
                operation="get_transactions",
                message=(
                    "Transactions retrieved successfully."
                ),
                data=result,
                metadata={
                    "company_id":
                        company_id,

                    "limit":
                        limit,

                    "offset":
                        offset,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve transactions: {exc}"
            ) from exc

    def get_transaction(
        self,
        *,
        transaction_id: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve a transaction by ID."""

        self._validate_identifier(
            transaction_id,
            "transaction_id",
        )

        if company_id is not None:
            self._validate_company_id(
                company_id
            )

        self._authorize(
            user_id=user_id,
            action="read_transaction",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.transaction_repository,
            "transaction_repository",
        )

        payload = {
            "transaction_id":
                transaction_id.strip(),

            "company_id":
                company_id,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_by_id",
                    "find_by_id",
                    "get_transaction",
                    "find",
                ),
                payload,
            )

            return self._success(
                operation="get_transaction",
                message=(
                    "Transaction retrieved successfully."
                ),
                data=result,
                metadata={
                    "transaction_id":
                        transaction_id,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve transaction: {exc}"
            ) from exc

    # ========================================================================
    # DOCUMENTS
    # ========================================================================

    def get_documents(
        self,
        *,
        company_id: str,
        document_type: Optional[
            str
        ] = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = DEFAULT_OFFSET,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve company documents."""

        self._validate_company_id(
            company_id
        )

        limit = self._validate_limit(
            limit
        )

        offset = self._validate_offset(
            offset
        )

        self._authorize(
            user_id=user_id,
            action="read_documents",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.document_repository,
            "document_repository",
        )

        payload = {
            "company_id":
                company_id,

            "document_type":
                document_type,

            "limit":
                limit,

            "offset":
                offset,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_company_documents",
                    "get_documents",
                    "list_by_company",
                    "find_by_company",
                    "list",
                ),
                payload,
            )

            return self._success(
                operation="get_documents",
                message=(
                    "Documents retrieved successfully."
                ),
                data=result,
                metadata={
                    "company_id":
                        company_id,

                    "limit":
                        limit,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve documents: {exc}"
            ) from exc

    def get_document(
        self,
        *,
        document_id: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve a single document."""

        self._validate_identifier(
            document_id,
            "document_id",
        )

        if company_id is not None:
            self._validate_company_id(
                company_id
            )

        self._authorize(
            user_id=user_id,
            action="read_document",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.document_repository,
            "document_repository",
        )

        payload = {
            "document_id":
                document_id.strip(),

            "company_id":
                company_id,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_by_id",
                    "find_by_id",
                    "get_document",
                    "find",
                ),
                payload,
            )

            return self._success(
                operation="get_document",
                message=(
                    "Document retrieved successfully."
                ),
                data=result,
                metadata={
                    "document_id":
                        document_id,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve document: {exc}"
            ) from exc

    # ========================================================================
    # CUSTOMERS
    # ========================================================================

    def get_customers(
        self,
        *,
        company_id: str,
        search: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = DEFAULT_OFFSET,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve customers for a company."""

        self._validate_company_id(
            company_id
        )

        limit = self._validate_limit(
            limit
        )

        offset = self._validate_offset(
            offset
        )

        if search is not None:
            search = self._validate_search(
                search
            )

        self._authorize(
            user_id=user_id,
            action="read_customers",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.customer_repository,
            "customer_repository",
        )

        payload = {
            "company_id":
                company_id,

            "search":
                search,

            "limit":
                limit,

            "offset":
                offset,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_company_customers",
                    "get_customers",
                    "list_by_company",
                    "find_by_company",
                    "list",
                ),
                payload,
            )

            return self._success(
                operation="get_customers",
                message=(
                    "Customers retrieved successfully."
                ),
                data=result,
                metadata={
                    "company_id":
                        company_id,

                    "limit":
                        limit,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve customers: {exc}"
            ) from exc

    def get_customer(
        self,
        *,
        customer_id: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve a customer by ID."""

        self._validate_identifier(
            customer_id,
            "customer_id",
        )

        if company_id is not None:
            self._validate_company_id(
                company_id
            )

        self._authorize(
            user_id=user_id,
            action="read_customer",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.customer_repository,
            "customer_repository",
        )

        payload = {
            "customer_id":
                customer_id.strip(),

            "company_id":
                company_id,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_by_id",
                    "find_by_id",
                    "get_customer",
                    "find",
                ),
                payload,
            )

            return self._success(
                operation="get_customer",
                message=(
                    "Customer retrieved successfully."
                ),
                data=result,
                metadata={
                    "customer_id":
                        customer_id,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve customer: {exc}"
            ) from exc

    # ========================================================================
    # ALERTS
    # ========================================================================

    def get_alerts(
        self,
        *,
        company_id: str,
        status: Optional[str] = None,
        severity: Optional[str] = None,
        alert_type: Optional[str] = None,
        limit: int = DEFAULT_LIMIT,
        offset: int = DEFAULT_OFFSET,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Retrieve persisted alerts.

        Alert creation/lifecycle operations should go through
        alert_tools.py / AlertService.
        """

        self._validate_company_id(
            company_id
        )

        limit = self._validate_limit(
            limit
        )

        offset = self._validate_offset(
            offset
        )

        self._authorize(
            user_id=user_id,
            action="read_alerts",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.alert_repository,
            "alert_repository",
        )

        payload = {
            "company_id":
                company_id,

            "status":
                status,

            "severity":
                severity,

            "alert_type":
                alert_type,

            "limit":
                limit,

            "offset":
                offset,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_company_alerts",
                    "get_alerts",
                    "list_by_company",
                    "find_by_company",
                    "list",
                ),
                payload,
            )

            return self._success(
                operation="get_alerts",
                message=(
                    "Alerts retrieved successfully."
                ),
                data=result,
                metadata={
                    "company_id":
                        company_id,

                    "limit":
                        limit,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve alerts: {exc}"
            ) from exc

    def get_alert(
        self,
        *,
        alert_id: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """Retrieve a single persisted alert."""

        self._validate_identifier(
            alert_id,
            "alert_id",
        )

        if company_id is not None:
            self._validate_company_id(
                company_id
            )

        self._authorize(
            user_id=user_id,
            action="read_alert",
            company_id=company_id,
        )

        repository = self._require_repository(
            self.alert_repository,
            "alert_repository",
        )

        payload = {
            "alert_id":
                alert_id.strip(),

            "company_id":
                company_id,
        }

        try:

            result = self._repository_call_with_payload(
                repository,
                (
                    "get_by_id",
                    "find_by_id",
                    "get_alert",
                    "find",
                ),
                payload,
            )

            return self._success(
                operation="get_alert",
                message=(
                    "Alert retrieved successfully."
                ),
                data=result,
                metadata={
                    "alert_id":
                        alert_id,
                },
            )

        except Exception as exc:

            raise DatabaseToolExecutionError(
                f"Failed to retrieve alert: {exc}"
            ) from exc

    # ========================================================================
    # GENERIC SAFE ENTITY LOOKUP
    # ========================================================================

    def get_entity(
        self,
        *,
        entity_type: str,
        entity_id: str,
        company_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> dict[str, Any]:
        """
        Generic safe entity lookup.

        This is NOT a generic SQL interface.

        Only explicitly supported entity types can be queried.
        """

        if not isinstance(
            entity_type,
            str,
        ):

            raise InvalidDatabaseToolInputError(
                "entity_type must be a string."
            )

        entity_type = (
            entity_type.strip().lower()
        )

        if (
            entity_type
            not in SUPPORTED_ENTITY_TYPES
        ):

            raise InvalidDatabaseToolInputError(
                "Unsupported entity type: "
                f"{entity_type}."
            )

        self._validate_identifier(
            entity_id,
            "entity_id",
        )

        if entity_type == "company":

            return self.get_company(
                company_id=entity_id,
                user_id=user_id,
            )

        if entity_type == "financial":

            if company_id is None:

                raise InvalidDatabaseToolInputError(
                    "company_id is required for "
                    "financial entity lookup."
                )

            records = self.get_financial_records(
                company_id=company_id,
                limit=1,
                user_id=user_id,
            )

            return records

        if entity_type == "transaction":

            return self.get_transaction(
                transaction_id=entity_id,
                company_id=company_id,
                user_id=user_id,
            )

        if entity_type == "document":

            return self.get_document(
                document_id=entity_id,
                company_id=company_id,
                user_id=user_id,
            )

        if entity_type == "customer":

            return self.get_customer(
                customer_id=entity_id,
                company_id=company_id,
                user_id=user_id,
            )

        if entity_type == "alert":

            return self.get_alert(
                alert_id=entity_id,
                company_id=company_id,
                user_id=user_id,
            )

        raise InvalidDatabaseToolInputError(
            f"Unsupported entity type: {entity_type}"
        )

    # ========================================================================
    # Repository Helpers
    # ========================================================================

    @staticmethod
    def _require_repository(
        repository: Any,
        name: str,
    ) -> Any:

        if repository is None:

            raise DatabaseToolConfigurationError(
                f"{name} is not configured."
            )

        return repository

    @staticmethod
    def _repository_call(
        repository: Any,
        method_names: Sequence[str],
        value: Any,
    ) -> Any:
        """
        Call a simple repository lookup.

        Supported signatures:
            method(value)
            method(id=value)
        """

        last_error: Optional[
            Exception
        ] = None

        for name in method_names:

            method = getattr(
                repository,
                name,
                None,
            )

            if method is None:
                continue

            try:
                return method(
                    value
                )

            except TypeError as exc:

                last_error = exc

                try:

                    return method(
                        id=value
                    )

                except TypeError as second_exc:

                    last_error = second_exc

        if last_error is not None:

            raise DatabaseToolConfigurationError(
                "Repository method signature mismatch: "
                f"{last_error}"
            )

        raise DatabaseToolConfigurationError(
            "No compatible repository method found."
        )

    @staticmethod
    def _repository_call_with_payload(
        repository: Any,
        method_names: Sequence[str],
        payload: Mapping[str, Any],
    ) -> Any:
        """
        Call a repository using keyword arguments or a payload.

        The repository remains the abstraction boundary.
        """

        last_error: Optional[
            Exception
        ] = None

        for name in method_names:

            method = getattr(
                repository,
                name,
                None,
            )

            if method is None:
                continue

            try:

                return method(
                    **dict(payload)
                )

            except TypeError as exc:

                last_error = exc

                try:

                    return method(
                        dict(payload)
                    )

                except TypeError as second_exc:

                    last_error = second_exc

        if last_error is not None:

            raise DatabaseToolConfigurationError(
                "Repository method signature mismatch: "
                f"{last_error}"
            )

        raise DatabaseToolConfigurationError(
            "No compatible repository method found."
        )

    @staticmethod
    def _repository_list(
        repository: Any,
        method_names: Sequence[str],
        *,
        limit: int,
        offset: int,
    ) -> Any:

        return DatabaseTools._repository_call_with_payload(
            repository,
            method_names,
            {
                "limit":
                    limit,

                "offset":
                    offset,
            },
        )

    # ========================================================================
    # Authorization
    # ========================================================================

    def _authorize(
        self,
        *,
        user_id: Optional[str],
        action: str,
        company_id: Optional[str] = None,
    ) -> None:
        """
        Optional authorization hook.

        In production, connect this to core.permissions.py.

        If no authorization service is supplied, repository-level
        authorization is assumed to be enforced elsewhere.
        """

        if self.authorization_service is None:
            return

        payload = {
            "user_id":
                user_id,

            "action":
                action,

            "company_id":
                company_id,
        }

        method = getattr(
            self.authorization_service,
            "authorize",
            None,
        )

        if method is None:

            method = getattr(
                self.authorization_service,
                "check_permission",
                None,
            )

        if method is None:

            raise DatabaseToolConfigurationError(
                "authorization_service must expose "
                "authorize() or check_permission()."
            )

        try:

            allowed = method(
                **payload
            )

        except TypeError:

            allowed = method(
                user_id,
                action,
                company_id,
            )

        if allowed is False:

            raise DatabaseAccessDeniedError(
                f"Database operation denied: {action}"
            )

    # ========================================================================
    # Validation
    # ========================================================================

    @staticmethod
    def _validate_company_id(
        company_id: str,
    ) -> None:

        DatabaseTools._validate_identifier(
            company_id,
            "company_id",
        )

    @staticmethod
    def _validate_identifier(
        value: str,
        field_name: str,
    ) -> None:

        if not isinstance(
            value,
            str,
        ):

            raise InvalidDatabaseToolInputError(
                f"{field_name} must be a string."
            )

        if not value.strip():

            raise InvalidDatabaseToolInputError(
                f"{field_name} cannot be empty."
            )

        if len(value.strip()) > 200:

            raise InvalidDatabaseToolInputError(
                f"{field_name} is too long."
            )

    @staticmethod
    def _validate_limit(
        limit: int,
    ) -> int:

        if isinstance(
            limit,
            bool,
        ):

            raise InvalidDatabaseToolInputError(
                "limit must be an integer."
            )

        try:

            limit = int(
                limit
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidDatabaseToolInputError(
                "limit must be an integer."
            ) from exc

        if limit < 1:

            raise InvalidDatabaseToolInputError(
                "limit must be >= 1."
            )

        if limit > MAX_LIMIT:

            raise InvalidDatabaseToolInputError(
                f"limit cannot exceed {MAX_LIMIT}."
            )

        return limit

    @staticmethod
    def _validate_offset(
        offset: int,
    ) -> int:

        if isinstance(
            offset,
            bool,
        ):

            raise InvalidDatabaseToolInputError(
                "offset must be an integer."
            )

        try:

            offset = int(
                offset
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidDatabaseToolInputError(
                "offset must be an integer."
            ) from exc

        if offset < 0:

            raise InvalidDatabaseToolInputError(
                "offset cannot be negative."
            )

        return offset

    @staticmethod
    def _validate_date(
        value: Optional[str],
        field_name: str,
    ) -> Optional[str]:

        if value is None:
            return None

        if not isinstance(
            value,
            str,
        ):

            raise InvalidDatabaseToolInputError(
                f"{field_name} must be YYYY-MM-DD."
            )

        value = value.strip()

        try:

            parsed = date.fromisoformat(
                value
            )

        except ValueError as exc:

            raise InvalidDatabaseToolInputError(
                f"{field_name} must be YYYY-MM-DD."
            ) from exc

        return parsed.isoformat()

    @staticmethod
    def _validate_date_range(
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> None:

        if (
            start_date is None
            or end_date is None
        ):
            return

        if start_date > end_date:

            raise InvalidDatabaseToolInputError(
                "start_date cannot be after end_date."
            )

    @staticmethod
    def _optional_number(
        value: Any,
        field_name: str,
    ) -> Optional[float]:

        if value is None:
            return None

        if isinstance(
            value,
            bool,
        ):

            raise InvalidDatabaseToolInputError(
                f"{field_name} must be numeric."
            )

        try:

            number = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidDatabaseToolInputError(
                f"{field_name} must be numeric."
            ) from exc

        if not (
            float("-inf")
            < number
            < float("inf")
        ):

            raise InvalidDatabaseToolInputError(
                f"{field_name} must be finite."
            )

        return number

    @staticmethod
    def _validate_search(
        search: str,
    ) -> str:

        search = search.strip()

        if not search:

            raise InvalidDatabaseToolInputError(
                "search cannot be empty."
            )

        if len(search) > 200:

            raise InvalidDatabaseToolInputError(
                "search query is too long."
            )

        return search

    # ========================================================================
    # Response Helpers
    # ========================================================================

    @staticmethod
    def _success(
        *,
        operation: str,
        message: str,
        data: Any,
        metadata: Optional[
            Mapping[str, Any]
        ] = None,
    ) -> dict[str, Any]:

        return DatabaseToolResult(
            success=True,
            operation=operation,
            message=message,
            data=data,
            metadata=metadata,
        ).to_dict()


# ============================================================================
# Standalone Functions
# ============================================================================


def get_company(
    company_repository: Any,
    *,
    company_id: str,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone company lookup."""

    return DatabaseTools(
        company_repository=company_repository,
    ).get_company(
        company_id=company_id,
        user_id=user_id,
    )


def get_financial_records(
    financial_repository: Any,
    *,
    company_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = DEFAULT_OFFSET,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone financial lookup."""

    return DatabaseTools(
        financial_repository=financial_repository,
    ).get_financial_records(
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset,
        user_id=user_id,
    )


def get_latest_financials(
    financial_repository: Any,
    *,
    company_id: str,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone latest-financial lookup."""

    return DatabaseTools(
        financial_repository=financial_repository,
    ).get_latest_financials(
        company_id=company_id,
        user_id=user_id,
    )


def get_transactions(
    transaction_repository: Any,
    *,
    company_id: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    transaction_type: Optional[str] = None,
    status: Optional[str] = None,
    min_amount: Optional[float] = None,
    max_amount: Optional[float] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = DEFAULT_OFFSET,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone transaction lookup."""

    return DatabaseTools(
        transaction_repository=transaction_repository,
    ).get_transactions(
        company_id=company_id,
        start_date=start_date,
        end_date=end_date,
        transaction_type=transaction_type,
        status=status,
        min_amount=min_amount,
        max_amount=max_amount,
        limit=limit,
        offset=offset,
        user_id=user_id,
    )


def get_documents(
    document_repository: Any,
    *,
    company_id: str,
    document_type: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = DEFAULT_OFFSET,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone document lookup."""

    return DatabaseTools(
        document_repository=document_repository,
    ).get_documents(
        company_id=company_id,
        document_type=document_type,
        limit=limit,
        offset=offset,
        user_id=user_id,
    )


def get_customers(
    customer_repository: Any,
    *,
    company_id: str,
    search: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = DEFAULT_OFFSET,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone customer lookup."""

    return DatabaseTools(
        customer_repository=customer_repository,
    ).get_customers(
        company_id=company_id,
        search=search,
        limit=limit,
        offset=offset,
        user_id=user_id,
    )


def get_alerts(
    alert_repository: Any,
    *,
    company_id: str,
    status: Optional[str] = None,
    severity: Optional[str] = None,
    alert_type: Optional[str] = None,
    limit: int = DEFAULT_LIMIT,
    offset: int = DEFAULT_OFFSET,
    user_id: Optional[str] = None,
) -> dict[str, Any]:
    """Standalone alert lookup."""

    return DatabaseTools(
        alert_repository=alert_repository,
    ).get_alerts(
        company_id=company_id,
        status=status,
        severity=severity,
        alert_type=alert_type,
        limit=limit,
        offset=offset,
        user_id=user_id,
    )


# ============================================================================
# Agent Tool Definitions
# ============================================================================


DATABASE_TOOL_DEFINITIONS = [
    {
        "name":
            "get_company",

        "description":
            "Retrieve company information by company ID.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "list_companies",

        "description":
            "List companies accessible to the current user.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_financial_records",

        "description":
            (
                "Retrieve financial records for a company, "
                "optionally filtered by date."
            ),

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_latest_financials",

        "description":
            "Retrieve the latest financial record for a company.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_transactions",

        "description":
            (
                "Retrieve company transactions with safe "
                "filters for dates, status and amount."
            ),

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_transaction",

        "description":
            "Retrieve a transaction by transaction ID.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_documents",

        "description":
            "Retrieve company documents.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_document",

        "description":
            "Retrieve a document by document ID.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_customers",

        "description":
            "Retrieve customers belonging to a company.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_customer",

        "description":
            "Retrieve a customer by customer ID.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_alerts",

        "description":
            "Retrieve persisted alerts for a company.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_alert",

        "description":
            "Retrieve a persisted alert by alert ID.",

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
    {
        "name":
            "get_entity",

        "description":
            (
                "Retrieve a supported entity by ID. "
                "Supported types are company, financial, "
                "transaction, document, customer and alert."
            ),

        "category":
            "database",

        "read_only":
            True,

        "requires_confirmation":
            False,
    },
]


# ============================================================================
# Serialization
# ============================================================================


def _serialize(
    value: Any,
) -> Any:

    if value is None:
        return None

    if isinstance(
        value,
        Enum,
    ):
        return value.value

    if isinstance(
        value,
        (datetime, date),
    ):
        return value.isoformat()

    if isinstance(
        value,
        Mapping,
    ):

        return {
            str(key):
                _serialize(item)
            for key, item
            in value.items()
        }

    if isinstance(
        value,
        (list, tuple, set),
    ):

        return [
            _serialize(item)
            for item
            in value
        ]

    if hasattr(
        value,
        "model_dump",
    ):

        return _serialize(
            value.model_dump()
        )

    if hasattr(
        value,
        "to_dict",
    ):

        return _serialize(
            value.to_dict()
        )

    if is_dataclass(
        value
    ):

        return _serialize(
            asdict(value)
        )

    if hasattr(
        value,
        "__dict__",
    ):

        return {
            str(key):
                _serialize(item)
            for key, item
            in vars(value).items()
            if not key.startswith("_")
        }

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):

        return value

    return str(value)


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    # Exceptions
    "DatabaseToolError",
    "InvalidDatabaseToolInputError",
    "DatabaseToolExecutionError",
    "DatabaseToolConfigurationError",
    "DatabaseAccessDeniedError",

    # Result
    "DatabaseToolResult",

    # Toolkit
    "DatabaseTools",

    # Standalone functions
    "get_company",
    "get_financial_records",
    "get_latest_financials",
    "get_transactions",
    "get_documents",
    "get_customers",
    "get_alerts",

    # Registry metadata
    "DATABASE_TOOL_DEFINITIONS",
]