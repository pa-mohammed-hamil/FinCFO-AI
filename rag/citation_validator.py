
"""
FinCo AI - Citation Validator

File:
    backend/app/rag/citation_validator.py

Purpose:
    Validate citations generated from retrieved RAG chunks.

Features:
    - Citation ID validation
    - Citation marker extraction
    - Unsupported citation detection
    - Citation coverage calculation
    - Duplicate citation detection
    - Source identity validation
    - Evidence validation
    - Financial claim validation
    - Citation report generation

Important:
    This module validates citation references and evidence
    availability. It does not prove that an LLM's wording
    is factually correct. Numeric and semantic verification
    should be handled by financial calculation services
    and a separate claim-evaluation layer.
"""

from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .citation_generator import (
    Citation,
    CitationResult,
    CitationSource,
)


# ============================================================
# Constants
# ============================================================

CITATION_MARKER_PATTERN = re.compile(
    r"\[(\d+)\]"
)

CITATION_ID_PATTERN = re.compile(
    r"^cite_[a-f0-9]{16}$"
)

CHUNK_ID_PATTERN = re.compile(
    r"^chunk_[a-f0-9]{16}$"
)

DEFAULT_MIN_EVIDENCE_LENGTH = 10

FINANCIAL_CLAIM_PATTERNS = [
    # Currency amounts
    r"(?:₹|Rs\.?|INR|\$|USD|€|EUR|£|GBP)\s?"
    r"\d[\d,]*(?:\.\d+)?",

    # Percentages
    r"\b\d+(?:\.\d+)?\s?%",

    # Financial metrics
    r"\b(?:revenue|profit|loss|expenses?|"
    r"cash flow|ebitda|ebit|assets?|liabilities?|"
    r"debt|margin|growth|income|sales|turnover)\b",

    # Financial periods
    r"\b(?:FY|Q[1-4]|financial year|fiscal year)"
    r"[\s-]?\d{2,4}\b",
]


# ============================================================
# Data Models
# ============================================================


@dataclass
class ValidationIssue:
    """
    Represents a single validation issue.
    """

    issue_type: str
    message: str
    severity: str = "ERROR"

    citation_number: Optional[int] = None
    citation_id: Optional[str] = None

    details: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert issue to dictionary.
        """

        return {
            "issue_type": self.issue_type,
            "message": self.message,
            "severity": self.severity,
            "citation_number": self.citation_number,
            "citation_id": self.citation_id,
            "details": self.details,
        }


@dataclass
class CitationValidationResult:
    """
    Result of citation validation.
    """

    is_valid: bool

    total_citations: int = 0
    valid_citations: int = 0
    invalid_citations: int = 0

    citation_coverage: float = 0.0

    issues: List[ValidationIssue] = field(
        default_factory=list
    )

    valid_citation_numbers: List[int] = field(
        default_factory=list
    )

    invalid_citation_numbers: List[int] = field(
        default_factory=list
    )

    unsupported_citation_numbers: List[int] = field(
        default_factory=list
    )

    cited_answer: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert validation result into a dictionary.
        """

        return {
            "is_valid": self.is_valid,
            "total_citations": self.total_citations,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "citation_coverage": self.citation_coverage,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
            "valid_citation_numbers": (
                self.valid_citation_numbers
            ),
            "invalid_citation_numbers": (
                self.invalid_citation_numbers
            ),
            "unsupported_citation_numbers": (
                self.unsupported_citation_numbers
            ),
            "cited_answer": self.cited_answer,
        }


@dataclass
class ClaimValidationResult:
    """
    Result of financial claim validation.
    """

    claim: str

    has_financial_content: bool

    has_citation: bool

    citation_numbers: List[int] = field(
        default_factory=list
    )

    is_supported: bool = False

    issues: List[ValidationIssue] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert claim validation result into a dictionary.
        """

        return {
            "claim": self.claim,
            "has_financial_content": (
                self.has_financial_content
            ),
            "has_citation": self.has_citation,
            "citation_numbers": self.citation_numbers,
            "is_supported": self.is_supported,
            "issues": [
                issue.to_dict()
                for issue in self.issues
            ],
        }


@dataclass
class FinancialClaimValidationReport:
    """
    Report for validating financial claims.
    """

    total_claims: int = 0
    supported_claims: int = 0
    unsupported_claims: int = 0

    claims: List[ClaimValidationResult] = field(
        default_factory=list
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert report into a dictionary.
        """

        return {
            "total_claims": self.total_claims,
            "supported_claims": self.supported_claims,
            "unsupported_claims": self.unsupported_claims,
            "claims": [
                claim.to_dict()
                for claim in self.claims
            ],
        }


# ============================================================
# Utility Functions
# ============================================================


def extract_citation_numbers(
    text: str,
) -> List[int]:
    """
    Extract numbered citation markers.

    Example:
        "Revenue increased [1] and profit fell [2]."

    Returns:
        [1, 2]
    """

    if not isinstance(text, str):
        return []

    numbers = [
        int(match.group(1))
        for match in CITATION_MARKER_PATTERN.finditer(text)
    ]

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(numbers))


def extract_citation_ids(
    citations: Sequence[Citation],
) -> Set[str]:
    """
    Extract citation IDs.
    """

    return {
        citation.citation_id
        for citation in citations
        if citation.citation_id
    }


def has_financial_content(
    text: str,
) -> bool:
    """
    Determine whether text contains financial terminology
    or financial numerical values.
    """

    if not isinstance(text, str):
        return False

    text_lower = text.lower()

    return any(
        re.search(pattern, text_lower)
        for pattern in FINANCIAL_CLAIM_PATTERNS
    )


def normalize_claim(
    claim: str,
) -> str:
    """
    Normalize claim text for validation.
    """

    if not isinstance(claim, str):
        return ""

    claim = re.sub(
        r"\s+",
        " ",
        claim,
    )

    return claim.strip()


def split_into_claims(
    answer: str,
) -> List[str]:
    """
    Split answer into sentence-like claims.

    This is a lightweight heuristic, not a full NLP
    claim extraction system.
    """

    if not isinstance(answer, str):
        return []

    answer = answer.strip()

    if not answer:
        return []

    # Remove sources section from claim analysis.
    answer = re.split(
        r"\n#{1,6}\s*(?:sources|references)\s*:?",
        answer,
        flags=re.IGNORECASE,
    )[0]

    claims = re.split(
        r"(?<=[.!?])\s+|\n+",
        answer,
    )

    return [
        normalize_claim(claim)
        for claim in claims
        if normalize_claim(claim)
    ]


def calculate_coverage(
    valid_count: int,
    total_count: int,
) -> float:
    """
    Calculate citation coverage percentage.

    Returns:
        Float between 0 and 100.
    """

    if total_count <= 0:
        return 0.0

    return round(
        (valid_count / total_count) * 100,
        2,
    )


# ============================================================
# Citation Validator
# ============================================================


class CitationValidator:
    """
    Validates citations against retrieved evidence.

    Validation checks:
        1. Citation IDs
        2. Citation numbering
        3. Source identity
        4. Evidence presence
        5. Citation references in answers
        6. Unsupported citation markers
        7. Duplicate citations
    """

    def __init__(
        self,
        min_evidence_length: int = DEFAULT_MIN_EVIDENCE_LENGTH,
        require_source_identity: bool = True,
        require_evidence: bool = True,
    ) -> None:

        if min_evidence_length < 0:
            raise ValueError(
                "min_evidence_length cannot be negative"
            )

        self.min_evidence_length = min_evidence_length
        self.require_source_identity = (
            require_source_identity
        )
        self.require_evidence = require_evidence

    # ========================================================
    # Main Validation
    # ========================================================

    def validate(
        self,
        citation_result: CitationResult,
        answer: Optional[str] = None,
    ) -> CitationValidationResult:
        """
        Validate generated citations.

        Args:
            citation_result:
                Result from CitationGenerator.

            answer:
                Optional final answer containing [1], [2], etc.

        Returns:
            CitationValidationResult
        """

        if not isinstance(
            citation_result,
            CitationResult,
        ):
            raise TypeError(
                "citation_result must be CitationResult"
            )

        citations = citation_result.citations

        issues: List[ValidationIssue] = []

        valid_numbers: List[int] = []
        invalid_numbers: List[int] = []

        seen_ids: Set[str] = set()

        for expected_number, citation in enumerate(
            citations,
            start=1,
        ):

            citation_issues = self._validate_citation(
                citation=citation,
                expected_number=expected_number,
                seen_ids=seen_ids,
            )

            issues.extend(citation_issues)

            if citation_issues:
                invalid_numbers.append(
                    citation.citation_number
                )
            else:
                valid_numbers.append(
                    citation.citation_number
                )

            if citation.citation_id:
                seen_ids.add(citation.citation_id)

        referenced_numbers: List[int] = []

        if answer is not None:
            referenced_numbers = extract_citation_numbers(
                answer
            )

            citation_numbers = {
                citation.citation_number
                for citation in citations
            }

            for number in referenced_numbers:

                if number not in citation_numbers:

                    issues.append(
                        ValidationIssue(
                            issue_type=(
                                "UNSUPPORTED_CITATION"
                            ),
                            message=(
                                f"Citation [{number}] "
                                "does not exist in "
                                "the citation list."
                            ),
                            severity="ERROR",
                            citation_number=number,
                        )
                    )

        total = len(citations)
        valid_count = len(valid_numbers)

        coverage = calculate_coverage(
            valid_count=valid_count,
            total_count=total,
        )

        return CitationValidationResult(
            is_valid=not any(
                issue.severity == "ERROR"
                for issue in issues
            ),
            total_citations=total,
            valid_citations=valid_count,
            invalid_citations=len(invalid_numbers),
            citation_coverage=coverage,
            issues=issues,
            valid_citation_numbers=valid_numbers,
            invalid_citation_numbers=invalid_numbers,
            unsupported_citation_numbers=[
                issue.citation_number
                for issue in issues
                if issue.issue_type
                == "UNSUPPORTED_CITATION"
                and issue.citation_number is not None
            ],
            cited_answer=answer,
        )

    # ========================================================
    # Validate Single Citation
    # ========================================================

    def _validate_citation(
        self,
        citation: Citation,
        expected_number: int,
        seen_ids: Set[str],
    ) -> List[ValidationIssue]:
        """
        Validate one citation.
        """

        issues: List[ValidationIssue] = []

        if not isinstance(citation, Citation):

            issues.append(
                ValidationIssue(
                    issue_type="INVALID_CITATION_OBJECT",
                    message="Citation is not a Citation object.",
                    severity="ERROR",
                )
            )

            return issues

        # Citation numbering.
        if citation.citation_number != expected_number:

            issues.append(
                ValidationIssue(
                    issue_type="INVALID_CITATION_NUMBER",
                    message=(
                        f"Expected citation number "
                        f"{expected_number}, got "
                        f"{citation.citation_number}."
                    ),
                    severity="ERROR",
                    citation_number=(
                        citation.citation_number
                    ),
                )
            )

        # Citation ID.
        if not citation.citation_id:

            issues.append(
                ValidationIssue(
                    issue_type="MISSING_CITATION_ID",
                    message="Citation ID is missing.",
                    severity="ERROR",
                    citation_number=(
                        citation.citation_number
                    ),
                )
            )

        elif not CITATION_ID_PATTERN.match(
            citation.citation_id
        ):

            issues.append(
                ValidationIssue(
                    issue_type="INVALID_CITATION_ID",
                    message=(
                        f"Invalid citation ID: "
                        f"{citation.citation_id}"
                    ),
                    severity="ERROR",
                    citation_number=(
                        citation.citation_number
                    ),
                    citation_id=citation.citation_id,
                )
            )

        # Duplicate citation ID.
        if citation.citation_id in seen_ids:

            issues.append(
                ValidationIssue(
                    issue_type="DUPLICATE_CITATION",
                    message=(
                        f"Duplicate citation ID: "
                        f"{citation.citation_id}"
                    ),
                    severity="ERROR",
                    citation_number=(
                        citation.citation_number
                    ),
                    citation_id=citation.citation_id,
                )
            )

        # Source identity.
        if self.require_source_identity:

            if not self._has_source_identity(
                citation.source
            ):

                issues.append(
                    ValidationIssue(
                        issue_type="MISSING_SOURCE_IDENTITY",
                        message=(
                            "Citation has no identifiable "
                            "document, source, or chunk."
                        ),
                        severity="ERROR",
                        citation_number=(
                            citation.citation_number
                        ),
                    )
                )

        # Evidence.
        if self.require_evidence:

            evidence = citation.evidence or ""

            if len(evidence.strip()) < (
                self.min_evidence_length
            ):

                issues.append(
                    ValidationIssue(
                        issue_type="INSUFFICIENT_EVIDENCE",
                        message=(
                            "Citation does not contain "
                            "enough supporting evidence."
                        ),
                        severity="ERROR",
                        citation_number=(
                            citation.citation_number
                        ),
                    )
                )

        return issues

    # ========================================================
    # Source Identity
    # ========================================================

    @staticmethod
    def _has_source_identity(
        source: CitationSource,
    ) -> bool:
        """
        Check whether source contains an identity.
        """

        if not isinstance(source, CitationSource):
            return False

        return any(
            [
                source.document_id,
                source.document_name,
                source.source,
                source.chunk_id,
            ]
        )

    # ========================================================
    # Validate Answer Citations
    # ========================================================

    def validate_answer_citations(
        self,
        answer: str,
        citation_result: CitationResult,
    ) -> CitationValidationResult:
        """
        Validate citations used in an answer.
        """

        return self.validate(
            citation_result=citation_result,
            answer=answer,
        )

    # ========================================================
    # Validate Financial Claims
    # ========================================================

    def validate_financial_claims(
        self,
        answer: str,
        citation_result: CitationResult,
    ) -> FinancialClaimValidationReport:
        """
        Validate financial claims in an answer.

        This checks whether financial claims have
        citation markers pointing to valid citations.

        It does NOT establish numerical truth.
        """

        claims = split_into_claims(answer)

        citation_numbers = {
            citation.citation_number
            for citation in citation_result.citations
        }

        report = FinancialClaimValidationReport()

        for claim in claims:

            if not has_financial_content(claim):
                continue

            numbers = extract_citation_numbers(
                claim
            )

            issues: List[ValidationIssue] = []

            if not numbers:

                issues.append(
                    ValidationIssue(
                        issue_type=(
                            "UNCITED_FINANCIAL_CLAIM"
                        ),
                        message=(
                            "Financial claim has "
                            "no citation marker."
                        ),
                        severity="WARNING",
                    )
                )

            invalid_numbers = [
                number
                for number in numbers
                if number not in citation_numbers
            ]

            if invalid_numbers:

                issues.append(
                    ValidationIssue(
                        issue_type=(
                            "INVALID_FINANCIAL_CITATION"
                        ),
                        message=(
                            "Financial claim references "
                            "missing citation numbers."
                        ),
                        severity="ERROR",
                        details={
                            "invalid_numbers": (
                                invalid_numbers
                            )
                        },
                    )
                )

            is_supported = (
                bool(numbers)
                and not invalid_numbers
            )

            claim_result = ClaimValidationResult(
                claim=claim,
                has_financial_content=True,
                has_citation=bool(numbers),
                citation_numbers=numbers,
                is_supported=is_supported,
                issues=issues,
            )

            report.claims.append(claim_result)
            report.total_claims += 1

            if is_supported:
                report.supported_claims += 1
            else:
                report.unsupported_claims += 1

        return report

    # ========================================================
    # Citation Coverage
    # ========================================================

    def calculate_answer_coverage(
        self,
        answer: str,
        citation_result: CitationResult,
    ) -> float:
        """
        Calculate percentage of financial claims
        containing valid citation markers.
        """

        report = self.validate_financial_claims(
            answer=answer,
            citation_result=citation_result,
        )

        return calculate_coverage(
            valid_count=report.supported_claims,
            total_count=report.total_claims,
        )

    # ========================================================
    # Citation Number Validation
    # ========================================================

    def validate_citation_numbers(
        self,
        answer: str,
        citations: Sequence[Citation],
    ) -> List[ValidationIssue]:
        """
        Find citation markers that do not exist
        in the available citation list.
        """

        issues: List[ValidationIssue] = []

        referenced = extract_citation_numbers(
            answer
        )

        available = {
            citation.citation_number
            for citation in citations
        }

        for number in referenced:

            if number not in available:

                issues.append(
                    ValidationIssue(
                        issue_type=(
                            "UNSUPPORTED_CITATION"
                        ),
                        message=(
                            f"Citation [{number}] "
                            "is not available."
                        ),
                        severity="ERROR",
                        citation_number=number,
                    )
                )

        return issues

    # ========================================================
    # Evidence Availability
    # ========================================================

    def validate_evidence_availability(
        self,
        citations: Sequence[Citation],
    ) -> List[ValidationIssue]:
        """
        Check whether citations have evidence.
        """

        issues: List[ValidationIssue] = []

        for citation in citations:

            if not citation.evidence:

                issues.append(
                    ValidationIssue(
                        issue_type="MISSING_EVIDENCE",
                        message=(
                            "Citation has no evidence text."
                        ),
                        severity="ERROR",
                        citation_number=(
                            citation.citation_number
                        ),
                        citation_id=citation.citation_id,
                    )
                )

        return issues


# ============================================================
# Convenience Functions
# ============================================================


def validate_citations(
    citation_result: CitationResult,
    answer: Optional[str] = None,
) -> CitationValidationResult:
    """
    Validate citations using default configuration.
    """

    validator = CitationValidator()

    return validator.validate(
        citation_result=citation_result,
        answer=answer,
    )


def validate_financial_claims(
    answer: str,
    citation_result: CitationResult,
) -> FinancialClaimValidationReport:
    """
    Validate financial claims using default configuration.
    """

    validator = CitationValidator()

    return validator.validate_financial_claims(
        answer=answer,
        citation_result=citation_result,
    )


# ============================================================
# Example Usage
# ============================================================


if __name__ == "__main__":

    from .citation_generator import (
        CitationGenerator,
    )

    retrieved_chunks = [
        {
            "chunk_id": "chunk_001",
            "text": (
                "Revenue increased to ₹10 crore "
                "during FY2025."
            ),
            "metadata": {
                "document_id": "doc_001",
                "document_name": "Annual Report 2025",
                "page_number": 12,
                "section": "Revenue",
                "source": "annual_report.pdf",
                "company_id": "company_001",
            },
            "score": 0.94,
        }
    ]

    generator = CitationGenerator()

    citation_result = generator.generate_citations(
        retrieved_chunks
    )

    answer = (
        "Revenue increased to ₹10 crore "
        "during FY2025. [1]"
    )

    validator = CitationValidator()

    validation_result = validator.validate(
        citation_result=citation_result,
        answer=answer,
    )

    print("Citation validation:")
    print(validation_result.to_dict())

    print("\nFinancial claim validation:")

    claim_report = validator.validate_financial_claims(
        answer=answer,
        citation_result=citation_result,
    )

    print(claim_report.to_dict())