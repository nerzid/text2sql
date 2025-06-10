import pytest
from unittest.mock import patch
from app.core.constants import TEXT_TO_SQL_PROMPT_TEMPLATE, IS_TOO_VAGUE_MESSAGE
from app.services.text_to_sql import (
    _build_table_str,
    get_relevant_headers,
    preprocess_text,
    text2sql,
)


class TestTextToSQL:
    @pytest.fixture
    def mock_collection(self):
        with patch("app.services.text_to_sql.collection") as mock_coll:
            yield mock_coll

    @pytest.fixture
    def mock_pipe(self):
        with patch("app.services.text_to_sql._pipe") as mock_pipe:
            mock_pipe.return_value = [
                {
                    "generated_text": "You are a SQL expert.\n\nGiven the question...\nquery=SELECT * FROM users"
                }
            ]
            yield mock_pipe

    @pytest.fixture
    def mock_disambiguator(self):
        with patch("app.services.text_to_sql.get_disambiguated_text") as mock_disamb:
            mock_disamb.return_value = {
                "disambiguated_text": "Show me all users",
                "is_too_vague": False,
            }
            yield mock_disamb

    def test_build_table_str(self):
        headers = ["id", "name", "email"]
        expected = "id (text) | name (text) | email (text)"
        assert _build_table_str(headers) == expected

    def test_get_relevant_headers_success(self, mock_collection):
        mock_collection.query.return_value = {"documents": [["id", "name", "email"]]}
        result = get_relevant_headers("Show me all users")
        assert result == ["id", "name", "email"]
        mock_collection.query.assert_called_once()

    def test_get_relevant_headers_empty(self, mock_collection):
        mock_collection.query.return_value = {"documents": [[]]}
        result = get_relevant_headers("Invalid query")
        assert result == []

    def test_get_relevant_headers_exception(self, mock_collection):
        mock_collection.query.side_effect = Exception("DB error")
        result = get_relevant_headers("Show me all users")
        assert result == []

    def test_preprocess_text_success(self, mock_disambiguator, mock_collection):
        mock_collection.query.return_value = {"documents": [["id", "name", "email"]]}
        result = preprocess_text("Show me all users")
        assert result == {
            "disambiguated_text": "Show me all users",
            "is_too_vague": False,
        }
        mock_disambiguator.assert_called_once()

    def test_preprocess_text_too_vague(self, mock_disambiguator, mock_collection):
        mock_collection.query.return_value = {"documents": [["id", "name", "email"]]}
        mock_disambiguator.return_value = {
            "disambiguated_text": "",
            "is_too_vague": True,
        }
        result = preprocess_text("vague query")
        assert result == {
            "disambiguated_text": "",
            "is_too_vague": True,
        }

    def test_preprocess_text_exception(self, mock_disambiguator, mock_collection):
        mock_collection.query.return_value = {"documents": [["id", "name", "email"]]}
        mock_disambiguator.side_effect = Exception("Processing error")
        query = "Show me all users"
        result = preprocess_text(query)
        assert result == query  # Should return original query on error

    def test_text2sql_success(self, mock_pipe, mock_collection):
        with patch("app.services.text_to_sql.load_model"):
            mock_collection.query.return_value = {
                "documents": [["id", "name", "email"]]
            }
            prompt_prefix = TEXT_TO_SQL_PROMPT_TEMPLATE.format(
                table_str="id (text) | name (text) | email (text)",
                question="Show me all user names",
            )
            mock_pipe.return_value = [
                {"generated_text": f"{prompt_prefix}\nquery=SELECT name FROM users"}
            ]
            result = text2sql("Show me all user names")
            assert result == "query=SELECT name FROM users"
