import pytest
from unittest.mock import Mock

from app.rag.chunking import chunk_text
from app.rag.metadata_filter import filter_by_metadata
from app.rag.citation_generator import generate_citation
from app.rag.retrieval_service import RetrievalService


# ---------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------

def test_chunk_text_returns_multiple_chunks():
    text = "Financial analysis. " * 200

    chunks = chunk_text(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) > 0 for chunk in chunks)


def test_chunk_text_empty_input():
    assert chunk_text("") == []


# ---------------------------------------------------------------------
# Metadata Filtering
# ---------------------------------------------------------------------

def test_filter_by_metadata_company():
    docs = [
        {"id": 1, "company": "FinCo", "text": "Revenue increased"},
        {"id": 2, "company": "OtherCo", "text": "Expenses increased"},
    ]

    result = filter_by_metadata(docs, {"company": "FinCo"})

    assert len(result) == 1
    assert result[0]["id"] == 1


# ---------------------------------------------------------------------
# Retrieval Service
# ---------------------------------------------------------------------

def test_retrieval_service_calls_retriever_and_reranker():
    retriever = Mock()
    retriever.retrieve.return_value = [
        {"id": 1, "text": "Revenue increased 20%", "score": 0.85}
    ]

    reranker = Mock()
    reranker.rerank.return_value = [
        {"id": 1, "text": "Revenue increased 20%", "score": 0.95}
    ]

    service = RetrievalService(retriever, reranker)

    results = service.retrieve("Why did revenue increase?")

    retriever.retrieve.assert_called_once_with(
        "Why did revenue increase?"
    )
    reranker.rerank.assert_called_once()

    assert results[0]["id"] == 1
    assert results[0]["score"] == pytest.approx(0.95)


def test_retrieval_service_empty_results():
    retriever = Mock()
    retriever.retrieve.return_value = []

    reranker = Mock()
    reranker.rerank.return_value = []

    service = RetrievalService(retriever, reranker)

    assert service.retrieve("Unknown question") == []


# ---------------------------------------------------------------------
# Citation Generation
# ---------------------------------------------------------------------

def test_generate_citation():
    metadata = {
        "source": "FY2025 Annual Report",
        "page": 42,
        "company": "FinCo",
    }

    citation = generate_citation(metadata)

    assert "FY2025 Annual Report" in citation
    assert "42" in citation
    assert "FinCo" in citation


# ---------------------------------------------------------------------
# Hybrid Retrieval Ordering
# ---------------------------------------------------------------------

def test_hybrid_retrieval_preserves_top_result():
    retriever = Mock()
    retriever.retrieve.return_value = [
        {"id": 5, "score": 0.91},
        {"id": 3, "score": 0.60},
    ]

    reranker = Mock()
    reranker.rerank.return_value = [
        {"id": 5, "score": 0.98},
        {"id": 3, "score": 0.72},
    ]

    service = RetrievalService(retriever, reranker)

    results = service.retrieve("gross margin")

    assert results[0]["id"] == 5
    assert results[0]["score"] > results[1]["score"]


# ---------------------------------------------------------------------
# Context Compression Behavior
# ---------------------------------------------------------------------

def test_context_size_reasonable():
    retriever = Mock()
    retriever.retrieve.return_value = [
        {"id": i, "text": "Financial information", "score": 1 - i * 0.01}
        for i in range(20)
    ]

    reranker = Mock()
    reranker.rerank.return_value = retriever.retrieve.return_value[:5]

    service = RetrievalService(retriever, reranker)

    results = service.retrieve("cash flow")

    assert len(results) <= 5


# ---------------------------------------------------------------------
# Metadata Preservation
# ---------------------------------------------------------------------

def test_retrieval_preserves_metadata():
    retriever = Mock()
    retriever.retrieve.return_value = [
        {
            "id": 10,
            "company": "FinCo",
            "page": 15,
            "score": 0.88,
            "text": "Operating expenses increased."
        }
    ]

    reranker = Mock()
    reranker.rerank.return_value = retriever.retrieve.return_value

    service = RetrievalService(retriever, reranker)

    result = service.retrieve("operating expenses")[0]

    assert result["company"] == "FinCo"
    assert result["page"] == 15