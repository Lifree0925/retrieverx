from uuid import uuid4
from app.core.fusion import reciprocal_rank_fusion
from app.models.retrieval import Candidate

def make_candidate(method):
    return Candidate(chunk_id=uuid4(), content="content", source=method, retrieval_method=method)

def test_rrf_deduplicates():
    first = make_candidate("bm25")
    second = first.model_copy(update={"retrieval_method": "vector"})
    result = reciprocal_rank_fusion([[first], [second]], k=60)
    assert len(result) == 1
    assert result[0].fusion_score > 0
    assert "bm25" in result[0].retrieval_method
    assert "vector" in result[0].retrieval_method
