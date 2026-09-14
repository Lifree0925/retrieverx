import pytest
from app.models.feedback import FeedbackEntry

def test_valid():
    assert FeedbackEntry(query="test", label="negative").label == "negative"

def test_invalid():
    with pytest.raises(ValueError):
        FeedbackEntry(query="test", label="unknown")
