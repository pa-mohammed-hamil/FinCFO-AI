
"""
FinCo AI - Citation Generator

File:
    backend/app/rag/citation_generator.py

Purpose:
    Generate reliable citations from retrieved RAG chunks.

Features:
    - Citation data models
    - Source and page references
    - Citation deduplication
    - Citation numbering
    - Citation formatting
    - Evidence attachment
    - Financial document support
    - Citation validation helpers

Usage:
    Retrieved chunks
        ↓
    CitationGenerator
        ↓
    Citation objects
        ↓
    LLM response with citations
"""

from __future__ import annotations

import hashlib
import re

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


# ============================================================
# Constants
# ============================================================

DEFAULT_CITATION_STYLE = "numbered"

SUPPORTED_CITATION_STYLES = {
    "numbered",
    "inline",
    "markdown",
    "footnote",
}

MAX_EVIDENCE_LENGTH = 500


# ============================================================
# Citation Data Models
# ============================================================


@dataclass
class CitationSource:
    """
    Represents the original source of evidence.

    A citation source identifies where the evidence came from.
    """

    document_id: Optional[str] = None
    document_name: Optional[str] = None
    document_type: Optional[str] = None

    page_number: Optional[int] = None
    section: Optional[str] = None

    source: Optional[str] = None
    company_id: Optional[str] = None

    chunk_id: Optional[str] = None

    start_char: Optional[int] = None
    end_char: Optional[int] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert source into a dictionary.
        """

        return {
            "document_id": self.document_id,
            "document_name": self.document_name,
            "document_type": self.document_type,
            "page_number": self.page_number,
            "section": self.section,
            "source": self.source,
            "company_id": self.company_id,
            "chunk_id": self.chunk_id,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "metadata": self.metadata,
        }


@dataclass
class Citation:
    """
    Represents a generated citation.

    Example:

        [1] Annual Report 2025, Page 12
    """

    citation_id: str
    citation_number: int

    source: CitationSource

    evidence: Optional[str] = None

    relevance_score: Optional[float] = None

    citation_text: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert citation into a dictionary.
        """

        return {
            "citation_id": self.citation_id,
            "citation_number": self.citation_number,
            "source": self.source.to_dict(),
            "evidence": self.evidence,
            "relevance_score": self.relevance_score,
            "citation_text": self.citation_text,
        }


@dataclass
class CitationResult:
    """
    Result returned by the citation generator.
    """

    citations: List[Citation] = field(
        default_factory=list
    )

    cited_answer: Optional[str] = None

    citation_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert result into a dictionary.
        """

        return {
            "citations": [
                citation.to_dict()
                for citation in self.citations
            ],
            "cited_answer": self.cited_answer,
            "citation_count": self.citation_count,
        }


# ============================================================
# Utility Functions
# ============================================================


def normalize_source_name(
    source: Optional[str],
) -> Optional[str]:
    """
    Normalize source name.
    """

    if not source:
        return None

    source = str(source).strip()

    if not source:
        return None

    return source


def truncate_evidence(
    evidence: Optional[str],
    max_length: int = MAX_EVIDENCE_LENGTH,
) -> Optional[str]:
    """
    Truncate evidence while preserving readable text.
    """

    if evidence is None:
        return None

    evidence = str(evidence).strip()

    if len(evidence) <= max_length:
        return evidence

    return evidence[:max_length].rstrip() + "..."


def clean_evidence(
    evidence: Optional[str],
) -> Optional[str]:
    """
    Clean extracted evidence text.
    """

    if evidence is None:
        return None

    evidence = str(evidence).strip()

    evidence = re.sub(
        r"[ \t]+",
        " ",
        evidence,
    )

    evidence = re.sub(
        r"\n{3,}",
        "\n\n",
        evidence,
    )

    return evidence.strip()


def generate_citation_id(
    source: CitationSource,
) -> str:
    """
    Generate a deterministic citation ID.

    Citation IDs are based on source identity.
    """

    raw_value = "|".join(
        [
            str(source.document_id or ""),
            str(source.document_name or ""),
            str(source.page_number or ""),
            str(source.section or ""),
            str(source.chunk_id or ""),
        ]
    )

    digest = hashlib.sha256(
        raw_value.encode("utf-8")
    ).hexdigest()[:16]

    return f"cite_{digest}"


# ============================================================
# Citation Generator
# ============================================================


class CitationGenerator:
    """
    Generates citations from retrieved document chunks.

    Expected chunk format:

        {
            "chunk_id": "chunk_001",
            "text": "...",
            "metadata": {
                "document_id": "doc_001",
                "document_name": "Annual Report",
                "page_number": 12,
                "section": "Revenue",
                "source": "annual_report.pdf",
                "company_id": "company_001"
            },
            "score": 0.91
        }
    """

    def __init__(
        self,
        citation_style: str = DEFAULT_CITATION_STYLE,
        max_evidence_length: int = MAX_EVIDENCE_LENGTH,
    ) -> None:

        if citation_style not in SUPPORTED_CITATION_STYLES:
            raise ValueError(
                f"Unsupported citation style: {citation_style}"
            )

        if max_evidence_length <= 0:
            raise ValueError(
                "max_evidence_length must be positive"
            )

        self.citation_style = citation_style
        self.max_evidence_length = max_evidence_length

    # --------------------------------------------------------
    # Public API
    # --------------------------------------------------------

    def generate_citations(
        self,
        retrieved_chunks: Sequence[Dict[str, Any]],
        max_citations: Optional[int] = None,
    ) -> CitationResult:
        """
        Generate citations from retrieved chunks.

        Args:
            retrieved_chunks:
                Chunks returned by the retriever.

            max_citations:
                Optional maximum number of citations.

        Returns:
            CitationResult
        """

        if max_citations is not None:
            if max_citations <= 0:
                raise ValueError(
                    "max_citations must be positive"
                )

        citations: List[Citation] = []
        seen_ids = set()

        for chunk in retrieved_chunks:

            citation = self._create_citation(
                chunk=chunk,
                citation_number=len(citations) + 1,
            )

            if citation is None:
                continue

            if citation.citation_id in seen_ids:
                continue

            seen_ids.add(citation.citation_id)
            citations.append(citation)

            if (
                max_citations is not None
                and len(citations) >= max_citations
            ):
                break

        return CitationResult(
            citations=citations,
            citation_count=len(citations),
        )

    # --------------------------------------------------------
    # Create Citation
    # --------------------------------------------------------

    def _create_citation(
        self,
        chunk: Dict[str, Any],
        citation_number: int,
    ) -> Optional[Citation]:
        """
        Create one citation from a retrieved chunk.
        """

        if not isinstance(chunk, dict):
            return None

        metadata = chunk.get("metadata") or {}

        if not isinstance(metadata, dict):
            metadata = {}

        text = chunk.get("text", "")

        if not isinstance(text, str):
            text = str(text)

        source = self._build_source(
            chunk=chunk,
            metadata=metadata,
        )

        if not self._has_source_identity(source):
            return None

        evidence = clean_evidence(text)

        evidence = truncate_evidence(
            evidence,
            max_length=self.max_evidence_length,
        )

        citation_id = generate_citation_id(source)

        relevance_score = self._extract_score(chunk)

        citation_text = self._format_citation(
            citation_number=citation_number,
            source=source,
        )

        return Citation(
            citation_id=citation_id,
            citation_number=citation_number,
            source=source,
            evidence=evidence,
            relevance_score=relevance_score,
            citation_text=citation_text,
        )

    # --------------------------------------------------------
    # Build Source
    # --------------------------------------------------------

    def _build_source(
        self,
        chunk: Dict[str, Any],
        metadata: Dict[str, Any],
    ) -> CitationSource:
        """
        Build citation source from chunk metadata.
        """

        return CitationSource(
            document_id=metadata.get("document_id"),
            document_name=metadata.get("document_name"),
            document_type=metadata.get("document_type"),
            page_number=self._extract_page_number(
                metadata
            ),
            section=metadata.get("section"),
            source=normalize_source_name(
                metadata.get("source")
            ),
            company_id=metadata.get("company_id"),
            chunk_id=chunk.get("chunk_id"),
            start_char=metadata.get("start_char"),
            end_char=metadata.get("end_char"),
            metadata={
                key: value
                for key, value in metadata.items()
                if key not in {
                    "document_id",
                    "document_name",
                    "document_type",
                    "page_number",
                    "section",
                    "source",
                    "company_id",
                    "start_char",
                    "end_char",
                }
            },
        )

    # --------------------------------------------------------
    # Page Number
    # --------------------------------------------------------

    @staticmethod
    def _extract_page_number(
        metadata: Dict[str, Any],
    ) -> Optional[int]:
        """
        Extract page number from metadata.

        Supports:
            page_number
            page
            page_index
        """

        page = metadata.get("page_number")

        if page is None:
            page = metadata.get("page")

        if page is None:
            page = metadata.get("page_index")

        if page is None:
            return None

        try:
            page = int(page)
        except (TypeError, ValueError):
            return None

        if page < 0:
            return None

        return page

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    @staticmethod
    def _extract_score(
        chunk: Dict[str, Any],
    ) -> Optional[float]:
        """
        Extract retrieval relevance score.
        """

        score = chunk.get("score")

        if score is None:
            score = chunk.get("relevance_score")

        if score is None:
            return None

        try:
            return float(score)
        except (TypeError, ValueError):
            return None

    # --------------------------------------------------------
    # Source Validation
    # --------------------------------------------------------

    @staticmethod
    def _has_source_identity(
        source: CitationSource,
    ) -> bool:
        """
        Determine whether a source has enough identity
        to generate a useful citation.
        """

        return any(
            [
                source.document_id,
                source.document_name,
                source.source,
                source.chunk_id,
            ]
        )

    # --------------------------------------------------------
    # Citation Formatting
    # --------------------------------------------------------

    def _format_citation(
        self,
        citation_number: int,
        source: CitationSource,
    ) -> str:
        """
        Format citation based on configured style.
        """

        source_name = (
            source.document_name
            or source.source
            or source.document_id
            or "Unknown source"
        )

        location_parts: List[str] = []

        if source.page_number is not None:
            location_parts.append(
                f"Page {source.page_number}"
            )

        if source.section:
            location_parts.append(
                f"Section: {source.section}"
            )

        location = ", ".join(location_parts)

        if location:
            description = f"{source_name}, {location}"
        else:
            description = source_name

        if self.citation_style == "numbered":
            return f"[{citation_number}] {description}"

        if self.citation_style == "inline":
            return f"[{citation_number}]"

        if self.citation_style == "markdown":
            return (
                f"[{citation_number}]: "
                f"{description}"
            )

        if self.citation_style == "footnote":
            return (
                f"[^{citation_number}]: "
                f"{description}"
            )

        return f"[{citation_number}] {description}"

    # ========================================================
    # Answer Citation
    # ========================================================

    def attach_citations_to_answer(
        self,
        answer: str,
        citations: Sequence[Citation],
    ) -> str:
        """
        Attach citation references to an answer.

        If the answer already contains citation markers,
        they are preserved.

        This method appends a Sources section.
        """

        if not isinstance(answer, str):
            raise TypeError(
                "answer must be a string"
            )

        answer = answer.strip()

        if not citations:
            return answer

        sources_section = self.format_sources(
            citations
        )

        if not answer:
            return sources_section

        return (
            f"{answer}\n\n"
            f"{sources_section}"
        )

    # ========================================================
    # Format Sources
    # ========================================================

    def format_sources(
        self,
        citations: Sequence[Citation],
    ) -> str:
        """
        Format citation sources into readable text.
        """

        if not citations:
            return ""

        lines = ["### Sources"]

        for citation in citations:
            lines.append(
                citation.citation_text
            )

        return "\n".join(lines)

    # ========================================================
    # Format Evidence
    # ========================================================

    def format_evidence(
        self,
        citations: Sequence[Citation],
    ) -> str:
        """
        Format citations with supporting evidence.
        """

        if not citations:
            return ""

        lines = ["### Evidence"]

        for citation in citations:

            lines.append(
                f"{citation.citation_text}\n"
                f"> {citation.evidence or 'No evidence available'}"
            )

        return "\n\n".join(lines)

    # ========================================================
    # Evidence Map
    # ========================================================

    def build_evidence_map(
        self,
        citations: Sequence[Citation],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Build mapping between citation IDs and evidence.
        """

        evidence_map: Dict[str, Dict[str, Any]] = {}

        for citation in citations:

            evidence_map[citation.citation_id] = {
                "citation_number": citation.citation_number,
                "citation_text": citation.citation_text,
                "evidence": citation.evidence,
                "source": citation.source.to_dict(),
                "relevance_score": citation.relevance_score,
            }

        return evidence_map

    # ========================================================
    # Financial Citation Helpers
    # ========================================================

    def generate_financial_citation(
        self,
        document_name: str,
        page_number: Optional[int] = None,
        section: Optional[str] = None,
        metric: Optional[str] = None,
        value: Optional[Any] = None,
        document_id: Optional[str] = None,
        company_id: Optional[str] = None,
    ) -> Citation:
        """
        Generate a citation for a financial metric.

        Example:
            Revenue = ₹10 crore
            Source: Annual Report, Page 12
        """

        source = CitationSource(
            document_id=document_id,
            document_name=document_name,
            page_number=page_number,
            section=section,
            company_id=company_id,
        )

        citation_id = generate_citation_id(source)

        evidence = None

        if metric is not None:

            if value is not None:
                evidence = (
                    f"{metric}: {value}"
                )
            else:
                evidence = metric

        citation_text = self._format_citation(
            citation_number=1,
            source=source,
        )

        return Citation(
            citation_id=citation_id,
            citation_number=1,
            source=source,
            evidence=evidence,
            citation_text=citation_text,
        )


# ============================================================
# Standalone Convenience Functions
# ============================================================


def generate_citations(
    retrieved_chunks: Sequence[Dict[str, Any]],
    max_citations: Optional[int] = None,
) -> CitationResult:
    """
    Generate citations from retrieved chunks.
    """

    generator = CitationGenerator()

    return generator.generate_citations(
        retrieved_chunks=retrieved_chunks,
        max_citations=max_citations,
    )


def attach_citations(
    answer: str,
    citations: Sequence[Citation],
) -> str:
    """
    Attach citations to an answer.
    """

    generator = CitationGenerator()

    return generator.attach_citations_to_answer(
        answer=answer,
        citations=citations,
    )


# ============================================================
# Example Usage
# ============================================================


if __name__ == "__main__":

    retrieved_chunks = [
        {
            "chunk_id": "chunk_001",
            "text": (
                "The company generated revenue of "
                "₹10 crore during FY2025."
            ),
            "metadata": {
                "document_id": "doc_001",
                "document_name": "Annual Report 2025",
                "document_type": "annual_report",
                "page_number": 12,
                "section": "Revenue",
                "source": "annual_report.pdf",
                "company_id": "company_001",
            },
            "score": 0.94,
        },
        {
            "chunk_id": "chunk_002",
            "text": (
                "Operating expenses increased by "
                "12 percent compared with FY2024."
            ),
            "metadata": {
                "document_id": "doc_001",
                "document_name": "Annual Report 2025",
                "document_type": "annual_report",
                "page_number": 15,
                "section": "Expenses",
                "source": "annual_report.pdf",
                "company_id": "company_001",
            },
            "score": 0.89,
        },
    ]

    generator = CitationGenerator(
        citation_style="numbered"
    )

    result = generator.generate_citations(
        retrieved_chunks=retrieved_chunks
    )

    answer = (
        "The company generated ₹10 crore in revenue "
        "during FY2025. Operating expenses increased "
        "by 12 percent."
    )

    cited_answer = generator.attach_citations_to_answer(
        answer=answer,
        citations=result.citations,
    )

    print(cited_answer)

    print("\nEvidence Map:")
    print(
        generator.build_evidence_map(
            result.citations
        )
    )