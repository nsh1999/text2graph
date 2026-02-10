"""PostgreSQL database integration module."""

import os
from typing import Optional, Any, List, Dict
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2.pool import SimpleConnectionPool
from dotenv import load_dotenv


class PostgreSQLConnection:
    """Manages PostgreSQL database connections and operations."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: int = 5432,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        min_connections: int = 1,
        max_connections: int = 10,
    ):
        """
        Initialize PostgreSQL connection manager.

        Args:
            host: Database host (default from DB_HOST env var)
            port: Database port (default 5432)
            database: Database name (default from DB_NAME env var)
            user: Database user (default from DB_USER env var)
            password: Database password (default from DB_PASSWORD env var)
            min_connections: Minimum connections in pool
            max_connections: Maximum connections in pool
        """
        load_dotenv()

        self.host = host or os.getenv("DB_HOST", "localhost")
        self.port = port or int(os.getenv("DB_PORT", 5432))
        self.database = database or os.getenv("DB_NAME")
        self.user = user or os.getenv("DB_USER")
        self.password = password or os.getenv("DB_PASSWORD")
        print(self.password)

        if not self.database or not self.user:
            raise ValueError("Database name and user are required")

        self.pool: Optional[SimpleConnectionPool] = None
        self._initialize_pool(min_connections, max_connections)

    def _initialize_pool(self, min_size: int, max_size: int) -> None:
        """Initialize psycopg2 connection pool."""
        try:
            self.pool = SimpleConnectionPool(
                min_size,
                max_size,
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
            )
        except Exception as e:
            raise ConnectionError(f"Failed to create connection pool: {e}")

    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.

        Yields:
            psycopg2 connection object
        """
        if not self.pool:
            raise RuntimeError("Connection pool not initialized")

        conn = None
        try:
            conn = self.pool.getconn()
            yield conn
            conn.commit()
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                self.pool.putconn(conn)

    def execute_query(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Execute a SELECT query and return results.

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            List of dictionaries representing rows
        """
        with self.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query, params or ())
                return cur.fetchall()

    def execute_scalar(
        self, query: str, params: Optional[tuple] = None
    ) -> Optional[Any]:
        """
        Execute a query that returns a single value.

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            Single value or None
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                result = cur.fetchone()
                return result[0] if result else None

    def execute_update(
        self, query: str, params: Optional[tuple] = None
    ) -> int:
        """
        Execute an INSERT, UPDATE, or DELETE query.

        Args:
            query: SQL query string
            params: Query parameters (optional)

        Returns:
            Number of affected rows
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(query, params or ())
                return cur.rowcount

    def bulk_insert(self, table: str, records: List[Dict[str, Any]]) -> int:
        """
        Perform a bulk insert of multiple records.

        Args:
            table: Table name
            records: List of dictionaries with column names as keys

        Returns:
            Number of inserted rows
        """
        if not records:
            return 0

        with self.get_connection() as conn:
            with conn.cursor() as cur:
                columns = list(records[0].keys())
                placeholders = ", ".join(["%s"] * len(columns))
                columns_str = ", ".join(columns)

                query = (
                    f"INSERT INTO {table} ({columns_str}) "
                    f"VALUES ({placeholders})"
                )

                for record in records:
                    values = tuple(record[col] for col in columns)
                    cur.execute(query, values)

                return len(records)

    def table_exists(self, table_name: str) -> bool:
        """
        Check if a table exists in the database.

        Args:
            table_name: Name of the table

        Returns:
            True if table exists, False otherwise
        """
        with self.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_name = %s
                    )
                    """,
                    (table_name,),
                )
                result = cur.fetchone()
                return result[0] if result else False

    def close(self) -> None:
        """Close all connections in the pool."""
        if self.pool:
            self.pool.closeall()
            self.pool = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Singleton instance for convenience
_db_instance: Optional[PostgreSQLConnection] = None


def get_db(
    host: Optional[str] = None,
    port: int = 5432,
    database: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> PostgreSQLConnection:
    """
    Get or create a database connection instance.

    Args:
        host: Database host
        port: Database port
        database: Database name
        user: Database user
        password: Database password

    Returns:
        PostgreSQLConnection instance
    """
    global _db_instance

    if _db_instance is None:
        _db_instance = PostgreSQLConnection(
            host=host, port=port, database=database, user=user, password=password
        )

    return _db_instance
