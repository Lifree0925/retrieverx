from app.core.query_classifier import QueryClassifier, QueryType

def test_semantic():
    assert QueryClassifier().classify("为什么最近销售下降") == QueryType.SEMANTIC

def test_exact_like_query():
    assert QueryClassifier().classify("SKU A123 的销售额是多少") in {
        QueryType.EXACT, QueryType.MIXED, QueryType.NUMERIC
    }
