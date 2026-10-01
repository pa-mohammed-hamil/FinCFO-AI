"""
Metadata filtering utilities for FinCo AI.

Supports:
- Exact metadata matching
- Inclusion and exclusion filters
- Comparison operators
- Date and numeric filtering
- Nested metadata fields
- AND / OR filter groups
- Tenant isolation
- Document-level filtering
- Retrieval result filtering

Example:

    filters = MetadataFilter(
        tenant_id="tenant_001",
        conditions={
            "document_type": "financial_report",
            "year": {"$gte": 2023},
        },
    )

    filtered_documents = filters.apply(documents)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class MetadataFilterError(ValueError):
    """Base exception for metadata filter errors."""


class InvalidFilterOperatorError(MetadataFilterError):
    """Raised when an unsupported filter operator is used."""


class MissingMetadataError(MetadataFilterError):
    """Raised when required metadata is missing."""


# ============================================================================
# Enums
# ============================================================================


class FilterOperator(str, Enum):
    """
    Supported metadata operators.
    """

    EQ = "$eq"
    NE = "$ne"
    GT = "$gt"
    GTE = "$gte"
    LT = "$lt"
    LTE = "$lte"
    IN = "$in"
    NIN = "$nin"
    EXISTS = "$exists"
    CONTAINS = "$contains"
    STARTSWITH = "$startswith"
    ENDSWITH = "$endswith"
    REGEX = "$regex"
    BETWEEN = "$between"
    ANY = "$any"
    ALL = "$all"


class FilterLogic(str, Enum):
    """
    Logical filter grouping.
    """

    AND = "and"
    OR = "or"


# ============================================================================
# Data models
# ============================================================================


@dataclass(frozen=True)
class MetadataCondition:
    """
    A single metadata condition.

    Examples:

        MetadataCondition("year", "$gte", 2023)
        MetadataCondition("document_type", "$eq", "annual_report")
        MetadataCondition("tags", "$contains", "revenue")
    """

    field: str
    operator: str | FilterOperator
    value: Any

    def __post_init__(self) -> None:
        if not self.field or not self.field.strip():
            raise MetadataFilterError(
                "Metadata condition field cannot be empty."
            )

        operator = (
            self.operator.value
            if isinstance(self.operator, FilterOperator)
            else str(self.operator).lower()
        )

        object.__setattr__(self, "field", self.field.strip())
        object.__setattr__(self, "operator", operator)


@dataclass
class MetadataFilterConfig:
    """
    Configuration for metadata filtering.
    """

    conditions: list[MetadataCondition] = field(default_factory=list)
    logic: FilterLogic = FilterLogic.AND
    case_sensitive: bool = False
    missing_field_matches: bool = False
    strict_types: bool = False


@dataclass
class FilterResult:
    """
    Result of evaluating metadata against a filter.
    """

    matched: bool
    metadata: Mapping[str, Any]
    failed_conditions: list[MetadataCondition] = field(
        default_factory=list
    )
    matched_conditions: list[MetadataCondition] = field(
        default_factory=list
    )

    @property
    def reason(self) -> str:
        if self.matched:
            return "All metadata conditions matched."

        if not self.failed_conditions:
            return "Metadata conditions did not match."

        return "; ".join(
            f"{condition.field} {condition.operator} {condition.value!r}"
            for condition in self.failed_conditions
        )


# ============================================================================
# General utilities
# ============================================================================


_MISSING = object()


def normalize_operator(
    operator: str | FilterOperator,
) -> str:
    """
    Normalize an operator to its string representation.
    """

    if isinstance(operator, FilterOperator):
        return operator.value

    value = str(operator).strip().lower()

    aliases = {
        "=": "$eq",
        "==": "$eq",
        "eq": "$eq",
        "!=": "$ne",
        "ne": "$ne",
        ">": "$gt",
        "gt": "$gt",
        ">=": "$gte",
        "gte": "$gte",
        "<": "$lt",
        "lt": "$lt",
        "<=": "$lte",
        "lte": "$lte",
        "in": "$in",
        "nin": "$nin",
        "not_in": "$nin",
        "exists": "$exists",
        "contains": "$contains",
        "startswith": "$startswith",
        "starts_with": "$startswith",
        "endswith": "$endswith",
        "ends_with": "$endswith",
        "regex": "$regex",
        "between": "$between",
        "any": "$any",
        "all": "$all",
    }

    return aliases.get(value, value)


def get_nested_value(
    metadata: Mapping[str, Any],
    field: str,
    *,
    default: Any = _MISSING,
) -> Any:
    """
    Read nested metadata using dot notation.

    Example:

        metadata = {
            "financial": {
                "year": 2024,
            }
        }

        get_nested_value(metadata, "financial.year")
    """

    if field in metadata:
        return metadata[field]

    current: Any = metadata

    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            if default is _MISSING:
                return _MISSING

            return default

        current = current[part]

    return current


def set_nested_value(
    metadata: dict[str, Any],
    field: str,
    value: Any,
) -> None:
    """
    Set a nested metadata field using dot notation.
    """

    parts = field.split(".")
    current = metadata

    for part in parts[:-1]:
        child = current.get(part)

        if not isinstance(child, dict):
            child = {}
            current[part] = child

        current = child

    current[parts[-1]] = value


def remove_nested_value(
    metadata: dict[str, Any],
    field: str,
) -> bool:
    """
    Remove a nested metadata field.

    Returns:
        True if a field was removed.
    """

    parts = field.split(".")
    current: Any = metadata

    for part in parts[:-1]:
        if not isinstance(current, Mapping) or part not in current:
            return False

        current = current[part]

    if not isinstance(current, dict):
        return False

    return current.pop(parts[-1], _MISSING) is not _MISSING


def clean_metadata(
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Convert metadata into a safe dictionary.
    """

    if metadata is None:
        return {}

    return dict(metadata)


def normalize_comparison_value(
    value: Any,
    *,
    case_sensitive: bool = False,
) -> Any:
    """
    Normalize values used in comparisons.
    """

    if isinstance(value, str) and not case_sensitive:
        return value.casefold()

    return value


def parse_datetime(value: Any) -> datetime | date | Any:
    """
    Parse common date/datetime strings.

    Non-date values are returned unchanged.
    """

    if isinstance(value, (datetime, date)):
        return value

    if not isinstance(value, str):
        return value

    text = value.strip()

    if not text:
        return value

    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        )
    except ValueError:
        pass

    try:
        return date.fromisoformat(text)
    except ValueError:
        return value


def comparable_values(
    actual: Any,
    expected: Any,
    *,
    case_sensitive: bool = False,
) -> tuple[Any, Any]:
    """
    Prepare values for comparison.
    """

    actual = parse_datetime(actual)
    expected = parse_datetime(expected)

    actual = normalize_comparison_value(
        actual,
        case_sensitive=case_sensitive,
    )
    expected = normalize_comparison_value(
        expected,
        case_sensitive=case_sensitive,
    )

    return actual, expected


# ============================================================================
# Operator evaluation
# ============================================================================


def evaluate_operator(
    actual: Any,
    operator: str | FilterOperator,
    expected: Any,
    *,
    case_sensitive: bool = False,
    strict_types: bool = False,
) -> bool:
    """
    Evaluate one metadata operator.

    Supported operators:

        $eq
        $ne
        $gt
        $gte
        $lt
        $lte
        $in
        $nin
        $exists
        $contains
        $startswith
        $endswith
        $regex
        $between
        $any
        $all
    """

    operator_value = normalize_operator(operator)

    if operator_value not in {
        item.value for item in FilterOperator
    }:
        raise InvalidFilterOperatorError(
            f"Unsupported metadata filter operator: {operator_value}"
        )

    exists = actual is not _MISSING

    if operator_value == "$exists":
        return exists is bool(expected)

    if not exists:
        return False

    if operator_value == "$eq":
        if strict_types and type(actual) is not type(expected):
            return False

        actual_value, expected_value = comparable_values(
            actual,
            expected,
            case_sensitive=case_sensitive,
        )

        return actual_value == expected_value

    if operator_value == "$ne":
        if strict_types and type(actual) is not type(expected):
            return True

        actual_value, expected_value = comparable_values(
            actual,
            expected,
            case_sensitive=case_sensitive,
        )

        return actual_value != expected_value

    if operator_value in {"$gt", "$gte", "$lt", "$lte"}:
        actual_value, expected_value = comparable_values(
            actual,
            expected,
            case_sensitive=case_sensitive,
        )

        try:
            if operator_value == "$gt":
                return actual_value > expected_value

            if operator_value == "$gte":
                return actual_value >= expected_value

            if operator_value == "$lt":
                return actual_value < expected_value

            return actual_value <= expected_value

        except TypeError:
            if strict_types:
                raise MetadataFilterError(
                    f"Cannot compare values: "
                    f"{actual_value!r} and {expected_value!r}"
                )

            return False

    if operator_value in {"$in", "$nin"}:
        if isinstance(expected, (str, bytes)) or not isinstance(
            expected,
            Iterable,
        ):
            raise MetadataFilterError(
                f"{operator_value} expects an iterable value."
            )

        expected_values = list(expected)

        actual_value = normalize_comparison_value(
            actual,
            case_sensitive=case_sensitive,
        )

        normalized_expected = [
            normalize_comparison_value(
                value,
                case_sensitive=case_sensitive,
            )
            for value in expected_values
        ]

        matched = actual_value in normalized_expected

        return matched if operator_value == "$in" else not matched

    if operator_value == "$contains":
        if isinstance(actual, str):
            actual_value = (
                actual
                if case_sensitive
                else actual.casefold()
            )
            expected_value = (
                str(expected)
                if case_sensitive
                else str(expected).casefold()
            )

            return expected_value in actual_value

        if isinstance(actual, Mapping):
            return expected in actual

        if isinstance(actual, Iterable):
            return expected in actual

        return False

    if operator_value == "$startswith":
        if not isinstance(actual, str):
            return False

        actual_value = (
            actual
            if case_sensitive
            else actual.casefold()
        )
        expected_value = (
            str(expected)
            if case_sensitive
            else str(expected).casefold()
        )

        return actual_value.startswith(expected_value)

    if operator_value == "$endswith":
        if not isinstance(actual, str):
            return False

        actual_value = (
            actual
            if case_sensitive
            else actual.casefold()
        )
        expected_value = (
            str(expected)
            if case_sensitive
            else str(expected).casefold()
        )

        return actual_value.endswith(expected_value)

    if operator_value == "$regex":
        if not isinstance(actual, str):
            return False

        flags = 0 if case_sensitive else re.IGNORECASE

        try:
            return re.search(
                str(expected),
                actual,
                flags=flags,
            ) is not None
        except re.error as exc:
            raise MetadataFilterError(
                f"Invalid regular expression: {expected!r}"
            ) from exc

    if operator_value == "$between":
        if not isinstance(expected, (list, tuple)) or len(expected) != 2:
            raise MetadataFilterError(
                "$between expects exactly two values."
            )

        lower_bound, upper_bound = expected

        actual_value, lower_bound = comparable_values(
            actual,
            lower_bound,
            case_sensitive=case_sensitive,
        )
        _, upper_bound = comparable_values(
            actual,
            upper_bound,
            case_sensitive=case_sensitive,
        )

        try:
            return lower_bound <= actual_value <= upper_bound
        except TypeError:
            if strict_types:
                raise MetadataFilterError(
                    f"Cannot apply $between to value: {actual!r}"
                )

            return False

    if operator_value in {"$any", "$all"}:
        if not isinstance(actual, Iterable) or isinstance(
            actual,
            (str, bytes, Mapping),
        ):
            return False

        expected_values = (
            list(expected)
            if isinstance(expected, Iterable)
            and not isinstance(expected, (str, bytes, Mapping))
            else [expected]
        )

        actual_values = list(actual)

        if operator_value == "$any":
            return any(
                evaluate_operator(
                    actual_value,
                    "$eq",
                    expected_value,
                    case_sensitive=case_sensitive,
                    strict_types=strict_types,
                )
                for actual_value in actual_values
                for expected_value in expected_values
            )

        return all(
            any(
                evaluate_operator(
                    actual_value,
                    "$eq",
                    expected_value,
                    case_sensitive=case_sensitive,
                    strict_types=strict_types,
                )
                for actual_value in actual_values
            )
            for expected_value in expected_values
        )

    return False


# ============================================================================
# Metadata filter
# ============================================================================


class MetadataFilter:
    """
    Main metadata filtering class.

    Supports simple conditions:

        MetadataFilter(
            conditions={
                "year": {"$gte": 2023},
                "document_type": "annual_report",
            }
        )

    Supports logical groups:

        MetadataFilter(
            any_of=[
                {"document_type": "annual_report"},
                {"document_type": "quarterly_report"},
            ]
        )
    """

    def __init__(
        self,
        conditions: (
            Mapping[str, Any]
            | Sequence[MetadataCondition]
            | None
        ) = None,
        *,
        tenant_id: str | None = None,
        logic: FilterLogic | str = FilterLogic.AND,
        case_sensitive: bool = False,
        missing_field_matches: bool = False,
        strict_types: bool = False,
        any_of: Sequence[Mapping[str, Any]] | None = None,
        all_of: Sequence[Mapping[str, Any]] | None = None,
        none_of: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.logic = FilterLogic(logic)
        self.case_sensitive = case_sensitive
        self.missing_field_matches = missing_field_matches
        self.strict_types = strict_types

        self.conditions = self._parse_conditions(conditions)

        self.any_of = [
            MetadataFilter(
                group,
                tenant_id=tenant_id,
                logic=FilterLogic.AND,
                case_sensitive=case_sensitive,
                missing_field_matches=missing_field_matches,
                strict_types=strict_types,
            )
            for group in (any_of or [])
        ]

        self.all_of = [
            MetadataFilter(
                group,
                tenant_id=tenant_id,
                logic=FilterLogic.AND,
                case_sensitive=case_sensitive,
                missing_field_matches=missing_field_matches,
                strict_types=strict_types,
            )
            for group in (all_of or [])
        ]

        self.none_of = [
            MetadataFilter(
                group,
                tenant_id=tenant_id,
                logic=FilterLogic.AND,
                case_sensitive=case_sensitive,
                missing_field_matches=missing_field_matches,
                strict_types=strict_types,
            )
            for group in (none_of or [])
        ]

    def _parse_conditions(
        self,
        conditions: (
            Mapping[str, Any]
            | Sequence[MetadataCondition]
            | None
        ),
    ) -> list[MetadataCondition]:
        if conditions is None:
            return []

        if isinstance(conditions, Mapping):
            parsed: list[MetadataCondition] = []

            for field_name, condition_value in conditions.items():
                if isinstance(condition_value, Mapping):
                    for operator, expected_value in condition_value.items():
                        parsed.append(
                            MetadataCondition(
                                field=field_name,
                                operator=operator,
                                value=expected_value,
                            )
                        )
                else:
                    parsed.append(
                        MetadataCondition(
                            field=field_name,
                            operator=FilterOperator.EQ,
                            value=condition_value,
                        )
                    )

            return parsed

        return list(conditions)

    def _evaluate_condition(
        self,
        metadata: Mapping[str, Any],
        condition: MetadataCondition,
    ) -> bool:
        actual = get_nested_value(
            metadata,
            condition.field,
            default=_MISSING,
        )

        if actual is _MISSING and self.missing_field_matches:
            actual = None

        return evaluate_operator(
            actual,
            condition.operator,
            condition.value,
            case_sensitive=self.case_sensitive,
            strict_types=self.strict_types,
        )

    def evaluate(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> FilterResult:
        """
        Evaluate metadata against this filter.
        """

        metadata = clean_metadata(metadata)

        if self.tenant_id is not None:
            actual_tenant_id = metadata.get("tenant_id")

            if actual_tenant_id != self.tenant_id:
                return FilterResult(
                    matched=False,
                    metadata=metadata,
                    failed_conditions=[
                        MetadataCondition(
                            field="tenant_id",
                            operator=FilterOperator.EQ,
                            value=self.tenant_id,
                        )
                    ],
                )

        matched_conditions: list[MetadataCondition] = []
        failed_conditions: list[MetadataCondition] = []

        for condition in self.conditions:
            matched = self._evaluate_condition(
                metadata,
                condition,
            )

            if matched:
                matched_conditions.append(condition)
            else:
                failed_conditions.append(condition)

        if self.logic == FilterLogic.AND:
            base_match = not failed_conditions
        else:
            base_match = bool(matched_conditions)

        if self.any_of:
            base_match = base_match and any(
                child.evaluate(metadata).matched
                for child in self.any_of
            )

        if self.all_of:
            base_match = base_match and all(
                child.evaluate(metadata).matched
                for child in self.all_of
            )

        if self.none_of:
            base_match = base_match and not any(
                child.evaluate(metadata).matched
                for child in self.none_of
            )

        return FilterResult(
            matched=base_match,
            metadata=metadata,
            failed_conditions=failed_conditions,
            matched_conditions=matched_conditions,
        )

    def matches(
        self,
        metadata: Mapping[str, Any] | None,
    ) -> bool:
        """
        Return True when metadata matches.
        """

        return self.evaluate(metadata).matched

    def apply(
        self,
        documents: Iterable[Any],
        *,
        metadata_getter: Any | None = None,
    ) -> list[Any]:
        """
        Filter arbitrary document objects.

        Supported document forms:
        - Mapping with a metadata key
        - Object with a metadata attribute
        - Object itself containing metadata fields

        Example:

            filter.apply(documents)
        """

        filtered: list[Any] = []

        for document in documents:
            metadata = self.extract_metadata(
                document,
                metadata_getter=metadata_getter,
            )

            if self.matches(metadata):
                filtered.append(document)

        return filtered

    @staticmethod
    def extract_metadata(
        document: Any,
        *,
        metadata_getter: Any | None = None,
    ) -> Mapping[str, Any]:
        if metadata_getter is not None:
            metadata = metadata_getter(document)
            return clean_metadata(metadata)

        if isinstance(document, Mapping):
            if "metadata" in document:
                return clean_metadata(document["metadata"])

            return clean_metadata(document)

        metadata = getattr(document, "metadata", None)

        if metadata is not None:
            return clean_metadata(metadata)

        return {}

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize this filter to a dictionary.
        """

        result: dict[str, Any] = {}

        for condition in self.conditions:
            field_name = condition.field
            operator = condition.operator
            value = condition.value

            existing = result.get(field_name)

            if existing is None:
                if operator == "$eq":
                    result[field_name] = value
                else:
                    result[field_name] = {
                        operator: value,
                    }
            else:
                if not isinstance(existing, dict):
                    existing = {
                        "$eq": existing,
                    }

                existing[operator] = value
                result[field_name] = existing

        if self.any_of:
            result["$or"] = [
                child.to_dict()
                for child in self.any_of
            ]

        if self.all_of:
            result["$and"] = [
                child.to_dict()
                for child in self.all_of
            ]

        if self.none_of:
            result["$nor"] = [
                child.to_dict()
                for child in self.none_of
            ]

        return result


# ============================================================================
# Retrieval result helpers
# ============================================================================


def filter_retrieval_results(
    results: Iterable[Any],
    metadata_filter: MetadataFilter,
) -> list[Any]:
    """
    Filter retrieval results using their metadata.
    """

    return metadata_filter.apply(results)


def filter_by_tenant(
    documents: Iterable[Any],
    tenant_id: str,
) -> list[Any]:
    """
    Apply tenant isolation to documents.

    This should be used as an additional safety layer in multi-tenant
    financial applications.
    """

    if not tenant_id or not str(tenant_id).strip():
        raise MetadataFilterError(
            "tenant_id cannot be empty."
        )

    metadata_filter = MetadataFilter(
        tenant_id=str(tenant_id),
    )

    return metadata_filter.apply(documents)


def build_metadata_filter(
    conditions: Mapping[str, Any] | None = None,
    *,
    tenant_id: str | None = None,
    case_sensitive: bool = False,
    strict_types: bool = False,
) -> MetadataFilter:
    """
    Convenience factory for MetadataFilter.
    """

    return MetadataFilter(
        conditions=conditions,
        tenant_id=tenant_id,
        case_sensitive=case_sensitive,
        strict_types=strict_types,
    )


__all__ = [
    "MetadataFilterError",
    "InvalidFilterOperatorError",
    "MissingMetadataError",
    "FilterOperator",
    "FilterLogic",
    "MetadataCondition",
    "MetadataFilterConfig",
    "FilterResult",
    "MetadataFilter",
    "normalize_operator",
    "get_nested_value",
    "set_nested_value",
    "remove_nested_value",
    "clean_metadata",
    "parse_datetime",
    "evaluate_operator",
    "filter_retrieval_results",
    "filter_by_tenant",
    "build_metadata_filter",
]