"""
FinCo AI - Fraud Transaction Preprocessing

File:
    backend/app/fraud/preprocessing.py

Purpose:
    Clean, validate, normalize, and enrich raw transaction data
    before fraud feature engineering and model inference.

Pipeline:

    Raw Transaction
          |
          v
    Validation
          |
          v
    Cleaning
          |
          v
    Type Normalization
          |
          v
    Missing Value Handling
          |
          v
    Timestamp Normalization
          |
          v
    Derived Transaction Metadata
          |
          v
    feature_engineering.py
          |
          v
    Fraud Models

Design goals:
    - Deterministic
    - Model independent
    - No LLM dependency
    - Safe for batch and online inference
    - Explainable transformations
    - Preserve original transaction fields
    - Avoid silently changing financial amounts
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
from copy import deepcopy
import math
import re


# ============================================================================
# Exceptions
# ============================================================================


class PreprocessingError(Exception):
    """Base exception for preprocessing errors."""


class InvalidTransactionError(PreprocessingError):
    """Raised when a transaction is invalid."""


class MissingTransactionFieldError(
    PreprocessingError
):
    """Raised when a required field is missing."""


class InvalidAmountError(
    PreprocessingError
):
    """Raised when transaction amount is invalid."""


class InvalidTimestampError(
    PreprocessingError
):
    """Raised when a transaction timestamp is invalid."""


class BatchPreprocessingError(
    PreprocessingError
):
    """Raised when batch preprocessing fails."""


# ============================================================================
# Constants
# ============================================================================


class TransactionField:
    """Canonical transaction field names."""

    TRANSACTION_ID = "transaction_id"
    COMPANY_ID = "company_id"
    CUSTOMER_ID = "customer_id"

    AMOUNT = "amount"
    CURRENCY = "currency"

    TRANSACTION_TYPE = "transaction_type"
    PAYMENT_METHOD = "payment_method"

    ACCOUNT_ID = "account_id"
    ACCOUNT_AGE_DAYS = "account_age_days"
    BALANCE = "balance"

    MERCHANT_ID = "merchant_id"
    MERCHANT_CATEGORY = "merchant_category"

    DEVICE_ID = "device_id"
    IP_ADDRESS = "ip_address"

    COUNTRY = "country"
    CITY = "city"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"

    TIMESTAMP = "timestamp"

    STATUS = "status"


class TransactionStatus:
    """Normalized transaction statuses."""

    COMPLETED = "COMPLETED"
    PENDING = "PENDING"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REVERSED = "REVERSED"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class PreprocessingConfig:
    """
    Configuration for transaction preprocessing.
    """

    required_fields: tuple[str, ...] = (
        TransactionField.AMOUNT,
    )

    default_currency: str = "USD"

    normalize_currency: bool = True

    normalize_categories: bool = True

    normalize_transaction_type: bool = True

    normalize_payment_method: bool = True

    parse_timestamps: bool = True

    reject_negative_amounts: bool = True

    reject_non_finite_amounts: bool = True

    clip_coordinates: bool = True

    remove_control_characters: bool = True

    strip_strings: bool = True

    uppercase_categories: bool = True

    preserve_original: bool = True

    add_processing_metadata: bool = True

    def validate(self) -> None:

        if not self.default_currency:
            raise ValueError(
                "default_currency cannot be empty."
            )

        if not self.required_fields:
            raise ValueError(
                "At least one required field "
                "must be configured."
            )


@dataclass
class PreprocessedTransaction:
    """
    Result of preprocessing one transaction.
    """

    transaction: Dict[str, Any]

    original: Optional[
        Dict[str, Any]
    ] = None

    warnings: List[str] = field(
        default_factory=list
    )

    transformations: List[str] = field(
        default_factory=list
    )

    processed_at: datetime = field(
        default_factory=lambda:
            datetime.now(timezone.utc)
    )

    valid: bool = True

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        """
        Return the cleaned transaction.

        Processing metadata is included inside
        `_preprocessing` rather than polluting
        business fields.
        """

        result = deepcopy(
            self.transaction
        )

        result["_preprocessing"] = {
            "valid": self.valid,
            "warnings": list(
                self.warnings
            ),
            "transformations": list(
                self.transformations
            ),
            "processed_at":
                self.processed_at.isoformat(),
        }

        return result


@dataclass
class BatchPreprocessingResult:
    """
    Result of preprocessing multiple transactions.
    """

    transactions: List[
        PreprocessedTransaction
    ] = field(
        default_factory=list
    )

    failed: List[
        Dict[str, Any]
    ] = field(
        default_factory=list
    )

    processed_count: int = 0

    failed_count: int = 0

    def to_dict(
        self,
    ) -> Dict[str, Any]:
        return {
            "transactions": [
                item.to_dict()
                for item in self.transactions
            ],
            "failed": list(
                self.failed
            ),
            "processed_count":
                self.processed_count,
            "failed_count":
                self.failed_count,
        }


# ============================================================================
# Main Preprocessor
# ============================================================================


class FraudTransactionPreprocessor:
    """
    Production-oriented transaction preprocessor.

    Responsibilities:

        1. Validate raw transaction.
        2. Normalize field names.
        3. Clean strings.
        4. Normalize numeric values.
        5. Validate amounts.
        6. Normalize categorical values.
        7. Parse timestamps.
        8. Normalize geographical values.
        9. Add safe derived metadata.
       10. Preserve original data when configured.

    It does NOT:

        - calculate fraud scores
        - train models
        - make fraud decisions
        - create alerts
        - call an LLM
    """

    FIELD_ALIASES = {
        "id": "transaction_id",
        "txn_id": "transaction_id",
        "transactionid": "transaction_id",

        "company": "company_id",
        "companyid": "company_id",

        "customer": "customer_id",
        "customerid": "customer_id",

        "value": "amount",
        "transaction_amount": "amount",
        "txn_amount": "amount",

        "currency_code": "currency",

        "type": "transaction_type",
        "txn_type": "transaction_type",

        "payment": "payment_method",
        "payment_type": "payment_method",

        "account": "account_id",
        "accountid": "account_id",

        "merchant": "merchant_id",
        "merchantid": "merchant_id",

        "device": "device_id",
        "deviceid": "device_id",

        "ip": "ip_address",
        "ipaddress": "ip_address",

        "time": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
    }

    TRANSACTION_TYPE_MAP = {
        "purchase": "PURCHASE",
        "buy": "PURCHASE",
        "payment": "PAYMENT",
        "transfer": "TRANSFER",
        "wire": "TRANSFER",
        "withdrawal": "WITHDRAWAL",
        "cash_withdrawal": "WITHDRAWAL",
        "deposit": "DEPOSIT",
        "refund": "REFUND",
        "refund_payment": "REFUND",
        "subscription": "SUBSCRIPTION",
        "bill_payment": "BILL_PAYMENT",
    }

    PAYMENT_METHOD_MAP = {
        "creditcard": "CREDIT_CARD",
        "credit_card": "CREDIT_CARD",
        "credit-card": "CREDIT_CARD",
        "debitcard": "DEBIT_CARD",
        "debit_card": "DEBIT_CARD",
        "debit-card": "DEBIT_CARD",
        "banktransfer": "BANK_TRANSFER",
        "bank_transfer": "BANK_TRANSFER",
        "bank-transfer": "BANK_TRANSFER",
        "wire": "BANK_TRANSFER",
        "cash": "CASH",
        "wallet": "DIGITAL_WALLET",
        "digital_wallet": "DIGITAL_WALLET",
        "digital-wallet": "DIGITAL_WALLET",
        "upi": "UPI",
        "paypal": "DIGITAL_WALLET",
    }

    STATUS_MAP = {
        "success": TransactionStatus.COMPLETED,
        "successful":
            TransactionStatus.COMPLETED,
        "complete":
            TransactionStatus.COMPLETED,
        "completed":
            TransactionStatus.COMPLETED,

        "pending":
            TransactionStatus.PENDING,

        "failed":
            TransactionStatus.FAILED,

        "cancelled":
            TransactionStatus.CANCELLED,
        "canceled":
            TransactionStatus.CANCELLED,

        "reversed":
            TransactionStatus.REVERSED,
    }

    def __init__(
        self,
        config: Optional[
            PreprocessingConfig
        ] = None,
    ) -> None:

        self.config = (
            config
            or PreprocessingConfig()
        )

        self.config.validate()

    # ========================================================================
    # SINGLE TRANSACTION
    # ========================================================================

    def process(
        self,
        transaction: Mapping[str, Any],
    ) -> PreprocessedTransaction:
        """
        Preprocess one transaction.
        """

        if not isinstance(
            transaction,
            Mapping,
        ):
            raise InvalidTransactionError(
                "Transaction must be a mapping."
            )

        if not transaction:
            raise InvalidTransactionError(
                "Transaction cannot be empty."
            )

        original = (
            deepcopy(
                dict(transaction)
            )
            if self.config.preserve_original
            else None
        )

        data = self._normalize_field_names(
            dict(transaction)
        )

        warnings: List[str] = []

        transformations: List[str] = []

        self._validate_required_fields(
            data
        )

        self._clean_strings(
            data,
            transformations,
        )

        self._normalize_amount(
            data,
            transformations,
        )

        self._normalize_currency(
            data,
            transformations,
        )

        self._normalize_categories(
            data,
            transformations,
        )

        self._normalize_numeric_fields(
            data,
            warnings,
            transformations,
        )

        self._normalize_coordinates(
            data,
            warnings,
            transformations,
        )

        if self.config.parse_timestamps:

            self._normalize_timestamp(
                data,
                warnings,
                transformations,
            )

        self._normalize_identifiers(
            data,
            transformations,
        )

        self._add_derived_metadata(
            data,
            transformations,
        )

        if self.config.add_processing_metadata:

            data["_preprocessing_version"] = (
                "1.0"
            )

        return PreprocessedTransaction(
            transaction=data,
            original=original,
            warnings=warnings,
            transformations=transformations,
        )

    # ========================================================================
    # BATCH PROCESSING
    # ========================================================================

    def process_batch(
        self,
        transactions: Iterable[
            Mapping[str, Any]
        ],
        fail_fast: bool = False,
    ) -> BatchPreprocessingResult:
        """
        Preprocess multiple transactions.

        By default one bad transaction does not
        stop the complete batch.
        """

        result = (
            BatchPreprocessingResult()
        )

        for index, transaction in enumerate(
            transactions
        ):

            try:

                processed = self.process(
                    transaction
                )

                result.transactions.append(
                    processed
                )

                result.processed_count += 1

            except Exception as exc:

                error = {
                    "index": index,
                    "transaction_id":
                        self._safe_transaction_id(
                            transaction
                        ),
                    "error_type":
                        type(exc).__name__,
                    "error":
                        str(exc),
                }

                result.failed.append(
                    error
                )

                result.failed_count += 1

                if fail_fast:

                    raise BatchPreprocessingError(
                        f"Batch preprocessing "
                        f"failed at index {index}."
                    ) from exc

        return result

    # ========================================================================
    # FIELD NAME NORMALIZATION
    # ========================================================================

    def _normalize_field_names(
        self,
        data: Dict[str, Any],
    ) -> Dict[str, Any]:

        normalized: Dict[
            str,
            Any,
        ] = {}

        for raw_key, value in data.items():

            key = self._normalize_key(
                raw_key
            )

            canonical = (
                self.FIELD_ALIASES.get(
                    key,
                    key,
                )
            )

            # Canonical key takes priority
            # over aliases.

            if (
                canonical in normalized
                and key != canonical
            ):
                continue

            normalized[
                canonical
            ] = value

        return normalized

    @staticmethod
    def _normalize_key(
        key: Any,
    ) -> str:

        text = str(
            key
        ).strip().lower()

        text = re.sub(
            r"[\s\-]+",
            "_",
            text,
        )

        text = re.sub(
            r"[^a-z0-9_]",
            "",
            text,
        )

        return text

    # ========================================================================
    # REQUIRED FIELDS
    # ========================================================================

    def _validate_required_fields(
        self,
        data: Mapping[str, Any],
    ) -> None:

        missing = []

        for field_name in (
            self.config.required_fields
        ):

            if (
                field_name not in data
                or data[field_name] is None
                or (
                    isinstance(
                        data[field_name],
                        str,
                    )
                    and not data[field_name].strip()
                )
            ):

                missing.append(
                    field_name
                )

        if missing:

            raise MissingTransactionFieldError(
                "Missing required transaction "
                f"fields: {', '.join(missing)}"
            )

    # ========================================================================
    # STRING CLEANING
    # ========================================================================

    def _clean_strings(
        self,
        data: Dict[str, Any],
        transformations: List[str],
    ) -> None:

        for key, value in list(
            data.items()
        ):

            if not isinstance(
                value,
                str,
            ):
                continue

            original = value

            if self.config.strip_strings:

                value = value.strip()

            if (
                self.config.remove_control_characters
            ):

                value = "".join(
                    character
                    for character in value
                    if (
                        character.isprintable()
                        or character in "\n\t"
                    )
                )

            if value != original:

                transformations.append(
                    f"cleaned_string:{key}"
                )

            data[key] = value

    # ========================================================================
    # AMOUNT
    # ========================================================================

    def _normalize_amount(
        self,
        data: Dict[str, Any],
        transformations: List[str],
    ) -> None:

        value = data.get(
            TransactionField.AMOUNT
        )

        if value is None:
            raise InvalidAmountError(
                "Transaction amount is missing."
            )

        if isinstance(
            value,
            str,
        ):

            cleaned = (
                value
                .replace(",", "")
                .replace("$", "")
                .replace("€", "")
                .replace("£", "")
                .replace("₹", "")
                .strip()
            )

            try:

                value = float(
                    cleaned
                )

            except ValueError as exc:

                raise InvalidAmountError(
                    f"Invalid transaction "
                    f"amount: {value!r}"
                ) from exc

            transformations.append(
                "amount_string_to_float"
            )

        try:

            amount = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:

            raise InvalidAmountError(
                "Transaction amount must "
                "be numeric."
            ) from exc

        if (
            self.config.reject_non_finite_amounts
            and not math.isfinite(amount)
        ):

            raise InvalidAmountError(
                "Transaction amount must "
                "be finite."
            )

        if (
            self.config.reject_negative_amounts
            and amount < 0
        ):

            raise InvalidAmountError(
                "Transaction amount "
                "cannot be negative."
            )

        data[
            TransactionField.AMOUNT
        ] = amount

    # ========================================================================
    # CURRENCY
    # ========================================================================

    def _normalize_currency(
        self,
        data: Dict[str, Any],
        transformations: List[str],
    ) -> None:

        currency = data.get(
            TransactionField.CURRENCY
        )

        if currency is None:

            data[
                TransactionField.CURRENCY
            ] = self.config.default_currency

            transformations.append(
                "currency_defaulted"
            )

            return

        if self.config.normalize_currency:

            currency = str(
                currency
            ).strip().upper()

            currency = (
                self._currency_alias(
                    currency
                )
            )

            data[
                TransactionField.CURRENCY
            ] = currency

            transformations.append(
                "currency_normalized"
            )

    @staticmethod
    def _currency_alias(
        currency: str,
    ) -> str:

        aliases = {
            "$": "USD",
            "US$": "USD",
            "USDOLLAR": "USD",

            "€": "EUR",
            "EURO": "EUR",

            "£": "GBP",
            "POUND": "GBP",

            "₹": "INR",
            "RUPEE": "INR",

            "YEN": "JPY",
            "¥": "JPY",
        }

        return aliases.get(
            currency,
            currency,
        )

    # ========================================================================
    # CATEGORIES
    # ========================================================================

    def _normalize_categories(
        self,
        data: Dict[str, Any],
        transformations: List[str],
    ) -> None:

        if not self.config.normalize_categories:
            return

        transaction_type = data.get(
            TransactionField.TRANSACTION_TYPE
        )

        if transaction_type is not None:

            normalized = (
                self._normalize_category_value(
                    transaction_type,
                    self.TRANSACTION_TYPE_MAP,
                )
            )

            data[
                TransactionField.TRANSACTION_TYPE
            ] = normalized

            transformations.append(
                "transaction_type_normalized"
            )

        payment_method = data.get(
            TransactionField.PAYMENT_METHOD
        )

        if payment_method is not None:

            normalized = (
                self._normalize_category_value(
                    payment_method,
                    self.PAYMENT_METHOD_MAP,
                )
            )

            data[
                TransactionField.PAYMENT_METHOD
            ] = normalized

            transformations.append(
                "payment_method_normalized"
            )

        status = data.get(
            TransactionField.STATUS
        )

        if status is not None:

            normalized = (
                self._normalize_category_value(
                    status,
                    self.STATUS_MAP,
                )
            )

            data[
                TransactionField.STATUS
            ] = normalized

            transformations.append(
                "status_normalized"
            )

        for key in (
            TransactionField.MERCHANT_CATEGORY,
            "merchant_type",
            "channel",
        ):

            if key not in data:
                continue

            value = data[key]

            if value is None:
                continue

            text = str(
                value
            ).strip()

            if self.config.uppercase_categories:

                text = re.sub(
                    r"[\s\-]+",
                    "_",
                    text.upper(),
                )

            data[key] = text

            transformations.append(
                f"category_normalized:{key}"
            )

    @staticmethod
    def _normalize_category_value(
        value: Any,
        aliases: Mapping[
            str,
            str,
        ],
    ) -> str:

        text = str(
            value
        ).strip().lower()

        compact = (
            text
            .replace(" ", "")
            .replace("-", "")
        )

        if compact in aliases:

            return aliases[
                compact
            ]

        if text in aliases:

            return aliases[
                text
            ]

        return re.sub(
            r"[\s\-]+",
            "_",
            text.upper(),
        )

    # ========================================================================
    # NUMERIC FIELDS
    # ========================================================================

    def _normalize_numeric_fields(
        self,
        data: Dict[str, Any],
        warnings: List[str],
        transformations: List[str],
    ) -> None:

        numeric_fields = (
            TransactionField.ACCOUNT_AGE_DAYS,
            TransactionField.BALANCE,
            TransactionField.LATITUDE,
            TransactionField.LONGITUDE,
            "velocity",
            "transaction_frequency",
            "merchant_frequency",
            "merchant_risk_score",
            "recent_amount_sum",
            "recent_amount_mean",
            "recent_amount_max",
            "recent_transaction_count",
        )

        for field_name in numeric_fields:

            if field_name not in data:
                continue

            value = data[field_name]

            if value is None:
                continue

            try:

                number = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ):

                warnings.append(
                    f"Invalid numeric value "
                    f"for field '{field_name}'."
                )

                continue

            if not math.isfinite(
                number
            ):

                warnings.append(
                    f"Non-finite numeric value "
                    f"for field '{field_name}'."
                )

                continue

            data[field_name] = number

            if not isinstance(
                value,
                float,
            ):

                transformations.append(
                    f"numeric_normalized:{field_name}"
                )

    # ========================================================================
    # COORDINATES
    # ========================================================================

    def _normalize_coordinates(
        self,
        data: Dict[str, Any],
        warnings: List[str],
        transformations: List[str],
    ) -> None:

        if not self.config.clip_coordinates:
            return

        latitude = data.get(
            TransactionField.LATITUDE
        )

        longitude = data.get(
            TransactionField.LONGITUDE
        )

        if latitude is not None:

            try:

                latitude = float(
                    latitude
                )

                if not (
                    -90 <= latitude <= 90
                ):

                    warnings.append(
                        "Latitude outside valid range."
                    )

                    latitude = max(
                        -90.0,
                        min(
                            90.0,
                            latitude,
                        ),
                    )

                    transformations.append(
                        "latitude_clipped"
                    )

                data[
                    TransactionField.LATITUDE
                ] = latitude

            except (
                TypeError,
                ValueError,
            ):
                warnings.append(
                    "Invalid latitude value."
                )

        if longitude is not None:

            try:

                longitude = float(
                    longitude
                )

                if not (
                    -180 <= longitude <= 180
                ):

                    warnings.append(
                        "Longitude outside valid range."
                    )

                    longitude = max(
                        -180.0,
                        min(
                            180.0,
                            longitude,
                        ),
                    )

                    transformations.append(
                        "longitude_clipped"
                    )

                data[
                    TransactionField.LONGITUDE
                ] = longitude

            except (
                TypeError,
                ValueError,
            ):
                warnings.append(
                    "Invalid longitude value."
                )

    # ========================================================================
    # TIMESTAMP
    # ========================================================================

    def _normalize_timestamp(
        self,
        data: Dict[str, Any],
        warnings: List[str],
        transformations: List[str],
    ) -> None:

        timestamp = data.get(
            TransactionField.TIMESTAMP
        )

        if timestamp is None:
            return

        try:

            parsed = self.parse_timestamp(
                timestamp
            )

            data[
                TransactionField.TIMESTAMP
            ] = parsed.isoformat()

            transformations.append(
                "timestamp_normalized"
            )

            # Useful numeric fields for
            # feature engineering.

            data[
                "timestamp_hour"
            ] = parsed.hour

            data[
                "timestamp_day_of_week"
            ] = parsed.weekday()

            data[
                "is_weekend"
            ] = parsed.weekday() >= 5

        except InvalidTimestampError as exc:

            warnings.append(
                str(exc)
            )

    # ========================================================================
    # TIMESTAMP PARSER
    # ========================================================================

    @staticmethod
    def parse_timestamp(
        value: Any,
    ) -> datetime:
        """
        Parse common timestamp formats.

        Naive timestamps are interpreted as UTC
        rather than guessing a local timezone.
        """

        if isinstance(
            value,
            datetime,
        ):

            parsed = value

        elif isinstance(
            value,
            (int, float),
        ):

            try:

                parsed = (
                    datetime.fromtimestamp(
                        float(value),
                        tz=timezone.utc,
                    )
                )

            except (
                ValueError,
                OverflowError,
                OSError,
            ) as exc:

                raise InvalidTimestampError(
                    f"Invalid timestamp: {value!r}"
                ) from exc

        elif isinstance(
            value,
            str,
        ):

            text = value.strip()

            if not text:

                raise InvalidTimestampError(
                    "Timestamp cannot be empty."
                )

            # ISO-8601.

            try:

                parsed = (
                    datetime.fromisoformat(
                        text.replace(
                            "Z",
                            "+00:00",
                        )
                    )
                )

            except ValueError:

                parsed = (
                    FraudTransactionPreprocessor
                    ._parse_common_datetime(
                        text
                    )
                )

        else:

            raise InvalidTimestampError(
                f"Unsupported timestamp type: "
                f"{type(value).__name__}"
            )

        if parsed.tzinfo is None:

            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        else:

            parsed = parsed.astimezone(
                timezone.utc
            )

        return parsed

    @staticmethod
    def _parse_common_datetime(
        text: str,
    ) -> datetime:

        formats = (
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y/%m/%d %H:%M:%S",
            "%Y/%m/%d %H:%M",

            "%d-%m-%Y %H:%M:%S",
            "%d-%m-%Y %H:%M",

            "%Y-%m-%d",
            "%Y/%m/%d",
            "%d-%m-%Y",
        )

        for fmt in formats:

            try:

                return datetime.strptime(
                    text,
                    fmt,
                )

            except ValueError:
                continue

        raise InvalidTimestampError(
            f"Unsupported timestamp format: "
            f"{text!r}"
        )

    # ========================================================================
    # IDENTIFIERS
    # ========================================================================

    def _normalize_identifiers(
        self,
        data: Dict[str, Any],
        transformations: List[str],
    ) -> None:

        identifier_fields = (
            TransactionField.TRANSACTION_ID,
            TransactionField.COMPANY_ID,
            TransactionField.CUSTOMER_ID,
            TransactionField.ACCOUNT_ID,
            TransactionField.MERCHANT_ID,
            TransactionField.DEVICE_ID,
        )

        for field_name in identifier_fields:

            if field_name not in data:
                continue

            value = data[field_name]

            if value is None:
                continue

            normalized = str(
                value
            ).strip()

            if normalized != value:

                transformations.append(
                    f"identifier_normalized:{field_name}"
                )

            data[field_name] = normalized

    # ========================================================================
    # DERIVED METADATA
    # ========================================================================

    def _add_derived_metadata(
        self,
        data: Dict[str, Any],
        transformations: List[str],
    ) -> None:

        amount = self._number(
            data.get(
                TransactionField.AMOUNT
            )
        )

        balance = self._number(
            data.get(
                TransactionField.BALANCE
            )
        )

        if (
            balance > 0
            and amount >= 0
        ):

            data[
                "amount_to_balance_ratio"
            ] = (
                amount / balance
            )

            transformations.append(
                "derived_amount_balance_ratio"
            )

        else:

            data[
                "amount_to_balance_ratio"
            ] = 0.0

        timestamp = data.get(
            TransactionField.TIMESTAMP
        )

        if timestamp:

            try:

                parsed = (
                    self.parse_timestamp(
                        timestamp
                    )
                )

                data[
                    "is_night"
                ] = (
                    parsed.hour < 6
                    or parsed.hour >= 22
                )

            except InvalidTimestampError:
                pass

        # International transaction flag.

        origin_country = data.get(
            "country"
        )

        destination_country = data.get(
            "destination_country"
        )

        if (
            origin_country
            and destination_country
        ):

            data[
                "is_international"
            ] = (
                str(
                    origin_country
                ).upper()
                != str(
                    destination_country
                ).upper()
            )

            transformations.append(
                "derived_international_flag"
            )

    # ========================================================================
    # HELPERS
    # ========================================================================

    @staticmethod
    def _number(
        value: Any,
        default: float = 0.0,
    ) -> float:

        try:

            number = float(
                value
            )

            if not math.isfinite(
                number
            ):
                return default

            return number

        except (
            TypeError,
            ValueError,
        ):
            return default

    @staticmethod
    def _safe_transaction_id(
        transaction: Any,
    ) -> Optional[str]:

        if not isinstance(
            transaction,
            Mapping,
        ):
            return None

        for key in (
            "transaction_id",
            "txn_id",
            "id",
        ):

            if key in transaction:

                value = transaction[key]

                if value is not None:

                    return str(
                        value
                    )

        return None


# ============================================================================
# Convenience Functions
# ============================================================================


def preprocess_transaction(
    transaction: Mapping[str, Any],
    config: Optional[
        PreprocessingConfig
    ] = None,
) -> PreprocessedTransaction:
    """
    Preprocess a single transaction.
    """

    processor = (
        FraudTransactionPreprocessor(
            config=config
        )
    )

    return processor.process(
        transaction
    )


def preprocess_transactions(
    transactions: Iterable[
        Mapping[str, Any]
    ],
    config: Optional[
        PreprocessingConfig
    ] = None,
    fail_fast: bool = False,
) -> BatchPreprocessingResult:
    """
    Preprocess a batch of transactions.
    """

    processor = (
        FraudTransactionPreprocessor(
            config=config
        )
    )

    return processor.process_batch(
        transactions,
        fail_fast=fail_fast,
    )


# ============================================================================
# Public API
# ============================================================================


__all__ = [
    "PreprocessingError",
    "InvalidTransactionError",
    "MissingTransactionFieldError",
    "InvalidAmountError",
    "InvalidTimestampError",
    "BatchPreprocessingError",
    "TransactionField",
    "TransactionStatus",
    "PreprocessingConfig",
    "PreprocessedTransaction",
    "BatchPreprocessingResult",
    "FraudTransactionPreprocessor",
    "preprocess_transaction",
    "preprocess_transactions",
]