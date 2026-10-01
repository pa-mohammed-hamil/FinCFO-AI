"""
FinCo AI - BM25 Lexical Retrieval
=================================

Production-oriented BM25 implementation for FinCo AI's RAG layer.

Responsibilities
----------------
- Tokenize financial/business text.
- Build a BM25 inverted index.
- Score documents using BM25 Okapi.
- Support top-k lexical retrieval.
- Support metadata filtering.
- Handle document IDs and chunks.
- Support incremental indexing.
- Remove and update documents.
- Return structured retrieval results.
- Provide explainable term-level scoring.
- Serialize/load the index for persistence.

Why BM25?
---------
BM25 is particularly useful for financial retrieval because exact terms
often matter:

    "EBITDA"
    "Q4 FY2026"
    "GST"
    "HDFC"
    "invoice 10492"
    "₹25,000"
    "12.5%"

Dense embeddings are useful for semantic similarity, while BM25 provides
strong exact-match and lexical retrieval.

Typical architecture
--------------------

                    User Query
                         |
              +----------+----------+
              |                     |
              v                     v
          BM25 Search          Vector Search
              |                     |
              +----------+----------+
                         |
                    Hybrid Ranker
                         |
                    RAG Context
                         |
                         v
                         LLM

Dependencies
------------
No mandatory external dependencies.

Optional:
    rank_bm25

This module intentionally implements BM25 internally so FinCo AI does not
need to depend on a third-party implementation for its core retrieval path.
"""

from __future__ import annotations

import json
import logging
import math
import pickle
import re
import threading
import unicodedata
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class BM25Error(Exception):
    """Base exception for BM25 retrieval errors."""


class BM25ConfigurationError(BM25Error):
    """Raised when BM25 configuration is invalid."""


class BM25IndexError(BM25Error):
    """Raised when an index operation fails."""


class BM25DocumentError(BM25Error):
    """Raised when a document is invalid."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BM25Config:
    """Configuration for the BM25 retriever."""

    # BM25 Okapi parameters.
    k1: float = 1.5
    b: float = 0.75

    # Tokenization.
    lowercase: bool = True
    normalize_unicode: bool = True
    remove_punctuation: bool = True

    # Token filtering.
    remove_stopwords: bool = False
    min_token_length: int = 1
    max_token_length: Optional[int] = None

    # Retrieval.
    default_top_k: int = 10
    minimum_score: Optional[float] = None

    # Length normalization.
    use_length_normalization: bool = True

    # Phrase handling.
    include_ngrams: bool = False
    ngram_min: int = 2
    ngram_max: int = 2

    # Safety.
    max_document_length: Optional[int] = 200_000
    max_query_length: Optional[int] = 10_000

    # Cache.
    enable_query_cache: bool = True
    max_query_cache_size: int = 512

    def __post_init__(self) -> None:
        if self.k1 < 0:
            raise BM25ConfigurationError(
                "BM25 k1 must be >= 0."
            )

        if not 0 <= self.b <= 1:
            raise BM25ConfigurationError(
                "BM25 b must be between 0 and 1."
            )

        if self.default_top_k <= 0:
            raise BM25ConfigurationError(
                "default_top_k must be greater than zero."
            )

        if self.min_token_length < 1:
            raise BM25ConfigurationError(
                "min_token_length must be >= 1."
            )

        if (
            self.max_token_length is not None
            and self.max_token_length < self.min_token_length
        ):
            raise BM25ConfigurationError(
                "max_token_length cannot be less than "
                "min_token_length."
            )

        if self.ngram_min < 1:
            raise BM25ConfigurationError(
                "ngram_min must be >= 1."
            )

        if self.ngram_max < self.ngram_min:
            raise BM25ConfigurationError(
                "ngram_max cannot be less than ngram_min."
            )


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class BM25Document:
    """
    A document/chunk indexed by BM25.

    `text` is the searchable content.

    `metadata` can contain:
        source
        page
        document_id
        chunk_id
        tenant_id
        document_type
        financial_year
        etc.
    """

    doc_id: str
    text: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    tokens: List[str] = field(
        default_factory=list
    )

    term_frequencies: Dict[str, int] = field(
        default_factory=dict
    )

    length: int = 0

    def to_dict(
        self,
        include_tokens: bool = False,
    ) -> Dict[str, Any]:
        data = {
            "doc_id": self.doc_id,
            "text": self.text,
            "metadata": self.metadata,
            "length": self.length,
        }

        if include_tokens:
            data["tokens"] = self.tokens
            data["term_frequencies"] = (
                self.term_frequencies
            )

        return data


@dataclass
class BM25Result:
    """One BM25 retrieval result."""

    doc_id: str
    score: float

    text: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    rank: int = 0

    matched_terms: List[str] = field(
        default_factory=list
    )

    term_scores: Dict[str, float] = field(
        default_factory=dict
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BM25SearchResult:
    """Complete BM25 search response."""

    query: str

    results: List[BM25Result] = field(
        default_factory=list
    )

    total_documents: int = 0

    matched_documents: int = 0

    search_time_ms: Optional[float] = None

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def top_result(self) -> Optional[BM25Result]:
        if not self.results:
            return None

        return self.results[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "results": [
                result.to_dict()
                for result in self.results
            ],
            "total_documents": self.total_documents,
            "matched_documents": self.matched_documents,
            "search_time_ms": self.search_time_ms,
            "metadata": self.metadata,
        }


@dataclass
class BM25Stats:
    """Index statistics."""

    document_count: int = 0
    vocabulary_size: int = 0

    total_tokens: int = 0
    average_document_length: float = 0.0

    indexed_terms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


class FinancialTokenizer:
    """
    Tokenizer optimized for financial/business text.

    The tokenizer intentionally preserves useful financial tokens such as:

        EBITDA
        FY2026
        Q4
        INR
        GST
        12.5%
        10,000
        invoice-102
        ABC/123

    It is not intended to be a linguistic NLP tokenizer.
    """

    TOKEN_PATTERN = re.compile(
        r"""
        (?:
            \d+(?:[.,]\d+)*%
        )
        |
        (?:
            [A-Za-z]+(?:[-_/][A-Za-z0-9]+)+
        )
        |
        (?:
            [A-Za-z]+[A-Za-z0-9]*
        )
        |
        (?:
            \d+(?:[.,]\d+)*
        )
        |
        (?:
            [₹$€£¥]\s*\d+(?:[.,]\d+)*
        )
        """,
        re.VERBOSE,
    )

    DEFAULT_STOPWORDS: Set[str] = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "with",
    }

    def __init__(
        self,
        config: BM25Config,
    ) -> None:
        self.config = config

    def tokenize(
        self,
        text: str,
    ) -> List[str]:
        """Tokenize text."""

        if text is None:
            return []

        text = str(text)

        if self.config.max_document_length is not None:
            text = text[
                : self.config.max_document_length
            ]

        if self.config.normalize_unicode:
            text = unicodedata.normalize(
                "NFKC",
                text,
            )

        if self.config.lowercase:
            text = text.lower()

        if self.config.remove_punctuation:
            # Preserve separators useful to financial identifiers,
            # while removing most surrounding punctuation.
            text = text.replace("–", "-")
            text = text.replace("—", "-")

        tokens = self.TOKEN_PATTERN.findall(text)

        result: List[str] = []

        for token in tokens:
            token = token.strip()

            if not token:
                continue

            if self.config.lowercase:
                token = token.lower()

            if (
                len(token)
                < self.config.min_token_length
            ):
                continue

            if (
                self.config.max_token_length is not None
                and len(token)
                > self.config.max_token_length
            ):
                continue

            if (
                self.config.remove_stopwords
                and token in self.DEFAULT_STOPWORDS
            ):
                continue

            result.append(token)

        if self.config.include_ngrams:
            result.extend(
                self._generate_ngrams(
                    result
                )
            )

        return result

    def _generate_ngrams(
        self,
        tokens: Sequence[str],
    ) -> List[str]:
        ngrams: List[str] = []

        for n in range(
            self.config.ngram_min,
            self.config.ngram_max + 1,
        ):
            if len(tokens) < n:
                continue

            for index in range(
                len(tokens) - n + 1
            ):
                ngrams.append(
                    "_".join(
                        tokens[
                            index:index + n
                        ]
                    )
                )

        return ngrams


# ---------------------------------------------------------------------------
# BM25 Retriever
# ---------------------------------------------------------------------------


class BM25Retriever:
    """
    BM25 Okapi lexical retriever.

    Supports:
        - add_document
        - add_documents
        - update_document
        - remove_document
        - clear
        - search
        - search_with_scores
        - metadata filtering
        - explanation
        - save/load
    """

    def __init__(
        self,
        config: Optional[BM25Config] = None,
    ) -> None:
        self.config = config or BM25Config()

        self.tokenizer = FinancialTokenizer(
            self.config
        )

        self._documents: Dict[
            str,
            BM25Document,
        ] = {}

        # Inverted index:
        #
        # term -> set(doc_id)
        self._inverted_index: Dict[
            str,
            Set[str],
        ] = defaultdict(set)

        # Document frequency:
        #
        # term -> number of documents containing term
        self._document_frequency: Dict[
            str,
            int,
        ] = Counter()

        self._document_count: int = 0
        self._total_tokens: int = 0

        self._average_document_length: float = 0.0

        self._dirty_statistics = True

        self._query_cache: Dict[
            Tuple[str, int],
            BM25SearchResult,
        ] = {}

        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def document_count(self) -> int:
        return self._document_count

    @property
    def vocabulary_size(self) -> int:
        return len(
            self._document_frequency
        )

    @property
    def average_document_length(self) -> float:
        self._refresh_statistics()

        return self._average_document_length

    @property
    def is_empty(self) -> bool:
        return self._document_count == 0

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_document(
        self,
        document: BM25Document,
    ) -> None:
        """Add one document to the index."""

        if not isinstance(
            document,
            BM25Document,
        ):
            raise BM25DocumentError(
                "document must be a BM25Document."
            )

        if not document.doc_id:
            raise BM25DocumentError(
                "Document ID cannot be empty."
            )

        if not isinstance(
            document.text,
            str,
        ):
            raise BM25DocumentError(
                "Document text must be a string."
            )

        with self._lock:
            if document.doc_id in self._documents:
                raise BM25DocumentError(
                    f"Document '{document.doc_id}' "
                    "already exists. Use update_document()."
                )

            indexed = self._prepare_document(
                document
            )

            self._documents[
                indexed.doc_id
            ] = indexed

            self._index_document(
                indexed
            )

            self._document_count += 1
            self._total_tokens += indexed.length

            self._dirty_statistics = True

            self._invalidate_cache()

    def add_documents(
        self,
        documents: Iterable[BM25Document],
    ) -> int:
        """
        Add multiple documents.

        Returns the number of documents added.
        """

        count = 0

        with self._lock:
            for document in documents:
                self.add_document(
                    document
                )
                count += 1

        return count

    def update_document(
        self,
        document: BM25Document,
    ) -> None:
        """Replace an existing document."""

        if document.doc_id not in self._documents:
            self.add_document(document)
            return

        with self._lock:
            self.remove_document(
                document.doc_id
            )
            self.add_document(document)

    def remove_document(
        self,
        doc_id: str,
    ) -> bool:
        """Remove a document from the index."""

        with self._lock:
            document = self._documents.pop(
                doc_id,
                None,
            )

            if document is None:
                return False

            for term in document.term_frequencies:
                posting = self._inverted_index.get(
                    term
                )

                if posting is None:
                    continue

                posting.discard(doc_id)

                if not posting:
                    self._inverted_index.pop(
                        term,
                        None,
                    )

            self._document_count -= 1
            self._total_tokens -= document.length

            self._dirty_statistics = True

            self._invalidate_cache()

            return True

    def clear(self) -> None:
        """Clear the entire index."""

        with self._lock:
            self._documents.clear()
            self._inverted_index.clear()
            self._document_frequency.clear()

            self._document_count = 0
            self._total_tokens = 0
            self._average_document_length = 0.0

            self._dirty_statistics = False

            self._invalidate_cache()

    # ------------------------------------------------------------------
    # Document preparation
    # ------------------------------------------------------------------

    def _prepare_document(
        self,
        document: BM25Document,
    ) -> BM25Document:
        tokens = self.tokenizer.tokenize(
            document.text
        )

        term_frequencies = Counter(tokens)

        return BM25Document(
            doc_id=str(document.doc_id),
            text=document.text,
            metadata=dict(
                document.metadata
            ),
            tokens=tokens,
            term_frequencies=dict(
                term_frequencies
            ),
            length=len(tokens),
        )

    def _index_document(
        self,
        document: BM25Document,
    ) -> None:
        for term in document.term_frequencies:
            self._inverted_index[
                term
            ].add(document.doc_id)

            self._document_frequency[
                term
            ] += 1

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: Optional[int] = None,
        filters: Optional[
            Mapping[str, Any]
        ] = None,
        filter_fn: Optional[
            Callable[[BM25Document], bool]
        ] = None,
        minimum_score: Optional[float] = None,
        explain: bool = False,
    ) -> BM25SearchResult:
        """
        Search the BM25 index.

        `filters` performs exact metadata matching.

        Example:

            filters={
                "document_type": "annual_report",
                "financial_year": "2026",
            }

        `filter_fn` can perform more complex filtering.
        """

        if not isinstance(query, str):
            raise BM25Error(
                "BM25 query must be a string."
            )

        query = query.strip()

        if not query:
            return BM25SearchResult(
                query=query,
                total_documents=self.document_count,
            )

        if self.config.max_query_length is not None:
            query = query[
                : self.config.max_query_length
            ]

        top_k = (
            top_k
            if top_k is not None
            else self.config.default_top_k
        )

        if top_k <= 0:
            raise BM25Error(
                "top_k must be greater than zero."
            )

        minimum_score = (
            minimum_score
            if minimum_score is not None
            else self.config.minimum_score
        )

        cache_key = (
            query,
            top_k,
        )

        if (
            self.config.enable_query_cache
            and filters is None
            and filter_fn is None
            and minimum_score
            == self.config.minimum_score
            and not explain
        ):
            cached = self._query_cache.get(
                cache_key
            )

            if cached is not None:
                return self._clone_result(
                    cached
                )

        query_tokens = self.tokenizer.tokenize(
            query
        )

        if not query_tokens:
            return BM25SearchResult(
                query=query,
                total_documents=self.document_count,
            )

        with self._lock:
            self._refresh_statistics()

            candidate_ids = self._candidate_documents(
                query_tokens
            )

            candidate_ids = self._apply_filters(
                candidate_ids,
                filters=filters,
                filter_fn=filter_fn,
            )

            scored: List[
                Tuple[
                    str,
                    float,
                    Dict[str, float],
                    List[str],
                ]
            ] = []

            for doc_id in candidate_ids:
                document = self._documents[
                    doc_id
                ]

                score, term_scores = (
                    self._score_document(
                        document,
                        query_tokens,
                    )
                )

                if (
                    minimum_score is not None
                    and score < minimum_score
                ):
                    continue

                matched_terms = [
                    term
                    for term in query_tokens
                    if term
                    in document.term_frequencies
                ]

                scored.append(
                    (
                        doc_id,
                        score,
                        term_scores,
                        matched_terms,
                    )
                )

            scored.sort(
                key=lambda item: item[1],
                reverse=True,
            )

            selected = scored[:top_k]

            results: List[BM25Result] = []

            for rank, (
                doc_id,
                score,
                term_scores,
                matched_terms,
            ) in enumerate(
                selected,
                start=1,
            ):
                document = self._documents[
                    doc_id
                ]

                results.append(
                    BM25Result(
                        doc_id=doc_id,
                        score=score,
                        text=document.text,
                        metadata=dict(
                            document.metadata
                        ),
                        rank=rank,
                        matched_terms=(
                            matched_terms
                            if explain
                            else []
                        ),
                        term_scores=(
                            term_scores
                            if explain
                            else {}
                        ),
                    )
                )

            result = BM25SearchResult(
                query=query,
                results=results,
                total_documents=(
                    self._document_count
                ),
                matched_documents=len(scored),
            )

            if (
                self.config.enable_query_cache
                and filters is None
                and filter_fn is None
                and minimum_score
                == self.config.minimum_score
                and not explain
            ):
                self._cache_result(
                    cache_key,
                    result,
                )

            return result

    def search_texts(
        self,
        query: str,
        documents: Sequence[str],
        *,
        top_k: Optional[int] = None,
    ) -> BM25SearchResult:
        """
        Convenience method for searching an in-memory text collection.

        A temporary retriever is created for this operation.
        """

        temporary = BM25Retriever(
            self.config
        )

        temporary.add_documents(
            BM25Document(
                doc_id=str(index),
                text=text,
            )
            for index, text in enumerate(
                documents
            )
        )

        return temporary.search(
            query,
            top_k=top_k,
        )

    # ------------------------------------------------------------------
    # Candidate generation
    # ------------------------------------------------------------------

    def _candidate_documents(
        self,
        query_tokens: Sequence[str],
    ) -> Set[str]:
        """
        Restrict scoring to documents containing at least one query term.

        This is substantially faster than scoring every document.
        """

        candidates: Set[str] = set()

        for term in set(query_tokens):
            candidates.update(
                self._inverted_index.get(
                    term,
                    set(),
                )
            )

        return candidates

    def _apply_filters(
        self,
        candidate_ids: Set[str],
        *,
        filters: Optional[
            Mapping[str, Any]
        ],
        filter_fn: Optional[
            Callable[[BM25Document], bool]
        ],
    ) -> Set[str]:
        if filters is None and filter_fn is None:
            return candidate_ids

        filtered: Set[str] = set()

        for doc_id in candidate_ids:
            document = self._documents[
                doc_id
            ]

            if filters is not None:
                if not self._matches_metadata_filters(
                    document,
                    filters,
                ):
                    continue

            if filter_fn is not None:
                try:
                    if not filter_fn(document):
                        continue
                except Exception as exc:
                    logger.warning(
                        "BM25 metadata filter failed "
                        "for document %s: %s",
                        doc_id,
                        exc,
                    )
                    continue

            filtered.add(doc_id)

        return filtered

    def _matches_metadata_filters(
        self,
        document: BM25Document,
        filters: Mapping[str, Any],
    ) -> bool:
        for key, expected in filters.items():
            actual = document.metadata.get(
                key
            )

            if isinstance(
                expected,
                (list, tuple, set, frozenset),
            ):
                if actual not in expected:
                    return False
            elif actual != expected:
                return False

        return True

    # ------------------------------------------------------------------
    # BM25 scoring
    # ------------------------------------------------------------------

    def _score_document(
        self,
        document: BM25Document,
        query_tokens: Sequence[str],
    ) -> Tuple[
        float,
        Dict[str, float],
    ]:
        """
        Calculate BM25 Okapi score.

        Formula:

            IDF(q) =
                log(
                    1 +
                    (N - df + 0.5)
                    / (df + 0.5)
                )

            score =
                IDF *
                (
                    tf * (k1 + 1)
                    /
                    (
                        tf +
                        k1 * (
                            1 - b +
                            b * dl / avgdl
                        )
                    )
                )
        """

        score = 0.0
        term_scores: Dict[str, float] = {}

        if (
            document.length == 0
            or self._document_count == 0
        ):
            return 0.0, term_scores

        avgdl = (
            self._average_document_length
            or 1.0
        )

        unique_query_terms = set(
            query_tokens
        )

        for term in unique_query_terms:
            tf = document.term_frequencies.get(
                term,
                0,
            )

            if tf == 0:
                continue

            df = self._document_frequency.get(
                term,
                0,
            )

            if df == 0:
                continue

            idf = math.log(
                1.0
                + (
                    self._document_count
                    - df
                    + 0.5
                )
                / (
                    df
                    + 0.5
                )
            )

            if self.config.use_length_normalization:
                normalization = (
                    1.0
                    - self.config.b
                    + self.config.b
                    * (
                        document.length
                        / avgdl
                    )
                )
            else:
                normalization = 1.0

            denominator = (
                tf
                + self.config.k1
                * normalization
            )

            term_score = (
                idf
                * (
                    tf
                    * (
                        self.config.k1
                        + 1.0
                    )
                )
                / denominator
            )

            term_scores[term] = term_score
            score += term_score

        return score, term_scores

    # ------------------------------------------------------------------
    # Explainability
    # ------------------------------------------------------------------

    def explain(
        self,
        query: str,
        doc_id: str,
    ) -> Dict[str, Any]:
        """
        Explain why a document matched a query.
        """

        if doc_id not in self._documents:
            raise BM25DocumentError(
                f"Document '{doc_id}' not found."
            )

        query_tokens = self.tokenizer.tokenize(
            query
        )

        with self._lock:
            self._refresh_statistics()

            document = self._documents[
                doc_id
            ]

            score, term_scores = (
                self._score_document(
                    document,
                    query_tokens,
                )
            )

            matched_terms = [
                term
                for term in query_tokens
                if term
                in document.term_frequencies
            ]

            return {
                "query": query,
                "doc_id": doc_id,
                "score": score,
                "matched_terms": matched_terms,
                "term_scores": term_scores,
                "document_length": document.length,
                "average_document_length": (
                    self._average_document_length
                ),
                "document_frequency": {
                    term: self._document_frequency.get(
                        term,
                        0,
                    )
                    for term in matched_terms
                },
            }

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def _refresh_statistics(self) -> None:
        if not self._dirty_statistics:
            return

        if self._document_count == 0:
            self._average_document_length = 0.0
        else:
            self._average_document_length = (
                self._total_tokens
                / self._document_count
            )

        self._dirty_statistics = False

    def stats(self) -> BM25Stats:
        """Return index statistics."""

        with self._lock:
            self._refresh_statistics()

            return BM25Stats(
                document_count=(
                    self._document_count
                ),
                vocabulary_size=(
                    self.vocabulary_size
                ),
                total_tokens=(
                    self._total_tokens
                ),
                average_document_length=(
                    self._average_document_length
                ),
                indexed_terms=(
                    len(self._inverted_index)
                ),
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(
        self,
        path: str | Path,
    ) -> None:
        """
        Persist the complete BM25 index.

        Pickle is used because the index contains Python-native sets,
        counters and document metadata.
        """

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            state = {
                "version": 1,
                "config": self.config,
                "documents": self._documents,
                "inverted_index": dict(
                    self._inverted_index
                ),
                "document_frequency": dict(
                    self._document_frequency
                ),
                "document_count": (
                    self._document_count
                ),
                "total_tokens": (
                    self._total_tokens
                ),
                "average_document_length": (
                    self._average_document_length
                ),
            }

            try:
                with target.open(
                    "wb"
                ) as handle:
                    pickle.dump(
                        state,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            except OSError as exc:
                raise BM25IndexError(
                    f"Unable to save BM25 index: "
                    f"{target}"
                ) from exc

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> "BM25Retriever":
        """Load a persisted BM25 index."""

        target = Path(path)

        if not target.exists():
            raise BM25IndexError(
                f"BM25 index does not exist: "
                f"{target}"
            )

        try:
            with target.open(
                "rb"
            ) as handle:
                state = pickle.load(handle)
        except (
            OSError,
            pickle.PickleError,
            EOFError,
        ) as exc:
            raise BM25IndexError(
                f"Unable to load BM25 index: "
                f"{target}"
            ) from exc

        if state.get("version") != 1:
            raise BM25IndexError(
                "Unsupported BM25 index version."
            )

        retriever = cls(
            config=state["config"]
        )

        retriever._documents = state[
            "documents"
        ]

        retriever._inverted_index = defaultdict(
            set,
            {
                term: set(doc_ids)
                for term, doc_ids
                in state[
                    "inverted_index"
                ].items()
            },
        )

        retriever._document_frequency = Counter(
            state["document_frequency"]
        )

        retriever._document_count = (
            state["document_count"]
        )

        retriever._total_tokens = (
            state["total_tokens"]
        )

        retriever._average_document_length = (
            state["average_document_length"]
        )

        retriever._dirty_statistics = False

        return retriever

    def save_json(
        self,
        path: str | Path,
    ) -> None:
        """
        Export a human-readable representation of the index.

        This is intended for diagnostics/debugging, not efficient reloads.
        """

        target = Path(path)

        target.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with self._lock:
            payload = {
                "version": 1,
                "config": asdict(
                    self.config
                ),
                "stats": self.stats().to_dict(),
                "documents": [
                    document.to_dict(
                        include_tokens=False
                    )
                    for document
                    in self._documents.values()
                ],
            }

        with target.open(
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(
                payload,
                handle,
                indent=2,
                ensure_ascii=False,
                default=str,
            )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    def _invalidate_cache(self) -> None:
        self._query_cache.clear()

    def _cache_result(
        self,
        key: Tuple[str, int],
        result: BM25SearchResult,
    ) -> None:
        if (
            self.config.max_query_cache_size
            <= 0
        ):
            return

        if (
            len(self._query_cache)
            >= self.config.max_query_cache_size
        ):
            oldest_key = next(
                iter(
                    self._query_cache
                )
            )

            self._query_cache.pop(
                oldest_key,
                None,
            )

        self._query_cache[
            key
        ] = self._clone_result(
            result
        )

    def _clone_result(
        self,
        result: BM25SearchResult,
    ) -> BM25SearchResult:
        return BM25SearchResult(
            query=result.query,
            results=[
                BM25Result(
                    doc_id=item.doc_id,
                    score=item.score,
                    text=item.text,
                    metadata=dict(
                        item.metadata
                    ),
                    rank=item.rank,
                    matched_terms=list(
                        item.matched_terms
                    ),
                    term_scores=dict(
                        item.term_scores
                    ),
                )
                for item in result.results
            ],
            total_documents=(
                result.total_documents
            ),
            matched_documents=(
                result.matched_documents
            ),
            search_time_ms=(
                result.search_time_ms
            ),
            metadata=dict(
                result.metadata
            ),
        )


# ---------------------------------------------------------------------------
# Factory / convenience functions
# ---------------------------------------------------------------------------


def create_bm25_retriever(
    config: Optional[BM25Config] = None,
) -> BM25Retriever:
    """Create a BM25 retriever."""

    return BM25Retriever(config)


def build_bm25_index(
    documents: Iterable[BM25Document],
    *,
    config: Optional[BM25Config] = None,
) -> BM25Retriever:
    """Build a BM25 index from documents."""

    retriever = BM25Retriever(config)

    retriever.add_documents(
        documents
    )

    return retriever


def bm25_search(
    retriever: BM25Retriever,
    query: str,
    *,
    top_k: Optional[int] = None,
    filters: Optional[
        Mapping[str, Any]
    ] = None,
) -> BM25SearchResult:
    """Convenience BM25 search function."""

    return retriever.search(
        query,
        top_k=top_k,
        filters=filters,
    )


__all__ = [
    "BM25Error",
    "BM25ConfigurationError",
    "BM25IndexError",
    "BM25DocumentError",
    "BM25Config",
    "BM25Document",
    "BM25Result",
    "BM25SearchResult",
    "BM25Stats",
    "FinancialTokenizer",
    "BM25Retriever",
    "create_bm25_retriever",
    "build_bm25_index",
    "bm25_search",
]