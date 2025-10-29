"""Test query processing"""
import pytest
from src.query_processing.pipeline import process_query

def test_process_query():
    """Test query processing"""
    query = "What is PTR?"
    dictionaries = {"acronyms": {"PTR": ["pupil-teacher ratio"]}}
    result = process_query(query, dictionaries)
    assert result is not None


