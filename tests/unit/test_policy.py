from app.core.query_classifier import QueryType
from app.core.retrieval_policy import RetrievalPolicy

def test_min_samples():
    policy = RetrievalPolicy(min_samples=20)
    old = policy.get(QueryType.EXACT)
    new = policy.update(QueryType.EXACT, 0.1, -0.1, 10)
    assert new.version == old.version

def test_version_update():
    policy = RetrievalPolicy(min_samples=1)
    old = policy.get(QueryType.EXACT)
    new = policy.update(QueryType.EXACT, 0.05, -0.05, 20)
    assert new.version == old.version + 1
