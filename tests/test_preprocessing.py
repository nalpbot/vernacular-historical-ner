import pytest

def test_token_alignment():
    sample_tokens = ["Devotional", "texts", "in", "Braj"]
    sample_tags = [0, 0, 0, 1]
    assert len(sample_tokens) == len(sample_tags), "Tokens and NER tags must have matching lengths."