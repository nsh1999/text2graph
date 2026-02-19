"""Unit tests for the PostgreSQL database module."""

import os
from unittest.mock import patch, MagicMock
import pytest

from text2graph.modules import PostgreSQLConnection


class TestPostgreSQLConnection:
    """Tests for PostgreSQLConnection class."""

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_init_with_env_vars(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test initialization with environment variables."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        # Should not raise - uses env vars
        db = PostgreSQLConnection()

        assert db.host == "localhost"
        assert db.database == "testdb"
        assert db.user == "testuser"
        assert db.port == 5432  # default
        mock_pool_class.assert_called_once()

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "db.example.com", "DB_NAME": "mydb", "DB_USER": "admin", "DB_PASSWORD": "secret"}, clear=True)
    def test_init_with_explicit_params(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test initialization with explicit parameters."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        db = PostgreSQLConnection(
            host="db.example.com",
            port=5433,
            database="mydb",
            user="admin",
            password="secret",
            min_connections=2,
            max_connections=20,
        )

        assert db.host == "db.example.com"
        assert db.port == 5433
        assert db.database == "mydb"
        assert db.user == "admin"
        assert db.password == "secret"
        mock_pool_class.assert_called_once_with(
            2, 20, host="db.example.com", port=5433,
            database="mydb", user="admin", password="secret"
        )

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost"}, clear=True)
    def test_init_missing_required_fields(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test initialization raises error when required fields are missing."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        # Missing database
        with pytest.raises(ValueError, match="Database name and user are required"):
            PostgreSQLConnection(host="localhost", user="test")

        # Missing user
        with pytest.raises(ValueError, match="Database name and user are required"):
            PostgreSQLConnection(host="localhost", database="testdb")

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_get_connection_context_manager(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test get_connection context manager."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        db = PostgreSQLConnection()
        db.pool = mock_pool

        with db.get_connection() as conn:
            assert conn == mock_conn

        mock_conn.commit.assert_called_once()
        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_get_connection_rollback_on_error(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test rollback on exception in context manager."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_pool_class.return_value = mock_pool

        db = PostgreSQLConnection()
        db.pool = mock_pool

        with pytest.raises(RuntimeError):
            with db.get_connection() as conn:
                raise RuntimeError("Test error")

        mock_conn.rollback.assert_called_once()
        mock_pool.putconn.assert_called_once_with(mock_conn)

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_execute_query(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test execute_query method."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool

        # Mock cursor.fetchall to return test data
        mock_cursor.fetchall.return_value = [
            {"id": 1, "name": "Item 1"},
            {"id": 2, "name": "Item 2"},
        ]

        db = PostgreSQLConnection()
        db.pool = mock_pool

        results = db.execute_query("SELECT * FROM items WHERE id = %s", (1,))

        assert len(results) == 2
        assert results[0]["id"] == 1
        mock_cursor.execute.assert_called_once_with("SELECT * FROM items WHERE id = %s", (1,))

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_execute_scalar(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test execute_scalar method."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool

        mock_cursor.fetchone.return_value = (42,)

        db = PostgreSQLConnection()
        db.pool = mock_pool

        result = db.execute_scalar("SELECT COUNT(*) FROM items")

        assert result == 42
        mock_cursor.fetchone.assert_called_once()

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_execute_scalar_no_result(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test execute_scalar returns None when no result."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool

        mock_cursor.fetchone.return_value = None

        db = PostgreSQLConnection()
        db.pool = mock_pool

        result = db.execute_scalar("SELECT * FROM empty_table")

        assert result is None

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_execute_update(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test execute_update method."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool

        mock_cursor.rowcount = 3

        db = PostgreSQLConnection()
        db.pool = mock_pool

        rows_affected = db.execute_update(
            "UPDATE items SET status = %s WHERE category = %s",
            ("active", "electronics")
        )

        assert rows_affected == 3
        mock_cursor.execute.assert_called_once()

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_bulk_insert(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test bulk_insert method."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool

        records = [
            {"name": "Item 1", "price": 10.99},
            {"name": "Item 2", "price": 20.99},
        ]

        db = PostgreSQLConnection()
        db.pool = mock_pool

        rows_inserted = db.bulk_insert("products", records)

        assert rows_inserted == 2
        assert mock_cursor.execute.call_count == 2

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_bulk_insert_empty_list(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test bulk_insert with empty list returns 0."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool

        db = PostgreSQLConnection()
        db.pool = mock_pool

        rows_inserted = db.bulk_insert("products", [])

        assert rows_inserted == 0
        mock_cursor.execute.assert_not_called()

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_table_exists(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test table_exists method."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool

        mock_cursor.fetchone.return_value = (True,)

        db = PostgreSQLConnection()
        db.pool = mock_pool

        exists = db.table_exists("users")

        assert exists is True
        mock_cursor.execute.assert_called_once()

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_table_exists_false(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test table_exists returns False when table doesn't exist."""
        mock_pool = MagicMock()
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.__enter__ = MagicMock(return_value=mock_cursor)
        mock_cursor.__exit__ = MagicMock(return_value=None)
        mock_pool.getconn.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_pool_class.return_value = mock_pool

        mock_cursor.fetchone.return_value = (False,)

        db = PostgreSQLConnection()
        db.pool = mock_pool

        exists = db.table_exists("nonexistent_table")

        assert exists is False

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_close(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test close method."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        db = PostgreSQLConnection()
        db.pool = mock_pool

        db.close()

        mock_pool.closeall.assert_called_once()
        assert db.pool is None

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_context_manager(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test PostgreSQLConnection as context manager."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        with PostgreSQLConnection() as db:
            assert db is not None

        mock_pool.closeall.assert_called_once()

    @patch("text2graph.modules.db.load_dotenv")
    @patch("text2graph.modules.db.SimpleConnectionPool")
    @patch.dict(os.environ, {"DB_HOST": "localhost", "DB_NAME": "testdb", "DB_USER": "testuser", "DB_PASSWORD": "testpass"}, clear=True)
    def test_singleton_get_db(self, mock_pool_class: MagicMock, mock_load_dotenv: MagicMock) -> None:
        """Test get_db singleton pattern."""
        mock_pool = MagicMock()
        mock_pool_class.return_value = mock_pool

        from text2graph.modules.db import _db_instance, get_db

        # Reset singleton
        _db_instance = None

        db1 = get_db()
        db2 = get_db()

        assert db1 is db2
        mock_pool_class.assert_called_once()
