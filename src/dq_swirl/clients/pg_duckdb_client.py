import os
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Generator, List, Optional, Type, Union

import psycopg
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool
from pydantic import BaseModel

from dq_swirl.utils.log_utils import get_custom_logger

logger = get_custom_logger()


@dataclass
class PGConfig:
    host: str = field(default_factory=lambda: os.getenv("POSTGRES_HOST", "localhost"))
    port: str | int = field(default_factory=lambda: os.getenv("POSTGRES_PORT", "5432"))
    user: str = field(
        default_factory=lambda: os.getenv("POSTGRES_USER", "app_developer")
    )
    password: str = field(
        default_factory=lambda: os.getenv("POSTGRES_PASSWORD", "password")
    )
    db: str = field(default_factory=lambda: os.getenv("POSTGRES_DB", "store"))

    def __post_init__(self):
        self.port = int(self.port)

    def __repr__(self) -> str:
        return (
            f"PostgresConfig(host='{self.host}', port='{self.port}', ",
            f"user='{self.user}', password='***', db='{self.db}')",
        )

    def __str__(self) -> str:
        return self.__repr__()

    def get_conn_str(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class PGDuckDBClient:
    def __init__(self, config: PGConfig, pool: Optional[ConnectionPool] = None) -> None:
        """_summary_

        :param config: _description_
        """
        self.config = config
        self.pool = pool
        if not self.pool:
            logger.info(f"Initializing Postgres ConnectionPool")
            self.pool = ConnectionPool(
                conninfo=self.config.get_conn_str(),
                min_size=2,
                max_size=4,
                configure=self._configure_connection,
                open=True,
            )

    def _configure_connection(self, conn: psycopg.Connection):
        """_summary_

        :param conn: _description_
        """
        old_autocommit = conn.autocommit

        try:
            conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS pg_duckdb;")
                cur.execute("SET duckdb.force_execution = true;")
        except Exception as e:
            logger.warning(f"Postgres Configuration Error: {e}")
        finally:
            conn.autocommit = old_autocommit

    def is_healthy(self, timeout: float = 2.0) -> bool:
        """_summary_

        :param timeout: _description_, defaults to 2.0
        :return: _description_
        """
        try:
            with self.pool.connection(timeout=timeout) as conn:
                conn.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    def create_table_from_model(
        self,
        model_class: Type[BaseModel],
        table_name: str = None,
        schema_name: str = "duckdb",
    ):
        """_summary_

        :param model_class: _description_
        :param schema_name: _description_, defaults to "duckdb"
        :param table_name: _description_, defaults to None
        """
        if not isinstance(model_class, type):
            model_class = model_class.__class__

        actual_table_name = table_name or model_class.__name__.lower()

        type_map = {
            str: "TEXT",
            int: "BIGINT",
            float: "DOUBLE PRECISION",
            bool: "BOOLEAN",
            dict: "JSONB",
            list: "JSONB",
            Any: "JSONB",
        }

        column_defs = []

        # .model_fields from the class directly
        for field_name, field_info in model_class.model_fields.items():
            python_type = field_info.annotation

            # handle optional/union types
            if hasattr(python_type, "__origin__") and python_type.__origin__ is Union:
                args = [t for t in python_type.__args__ if t is not type(None)]
                actual_type = args[0] if args else str
            else:
                actual_type = getattr(python_type, "__origin__", python_type)

            pg_type = type_map.get(actual_type, "TEXT")
            parts = [sql.Identifier(field_name), sql.SQL(pg_type)]

            # apply constraints
            if field_name.lower() in ("id"):
                parts.append(sql.SQL("PRIMARY KEY"))

            if field_info.is_required():
                parts.append(sql.SQL("NOT NULL"))

            column_defs.append(sql.SQL(" ").join(parts))

        # define identifiers for "schema"."table"
        schema_ident = sql.Identifier(schema_name)
        table_ident = sql.Identifier(schema_name, actual_table_name)

        with self.pool.connection() as conn:
            conn.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(schema_ident))
            create_query = sql.SQL(
                "CREATE TABLE IF NOT EXISTS {table} ({fields})"
            ).format(
                table=table_ident,
                fields=sql.SQL(", ").join(column_defs),
            )
            conn.execute(create_query)

            # column comments based on Pydantic field descriptions
            for field_name, field_info in model_class.model_fields.items():
                if field_info.description:
                    comment_query = sql.SQL(
                        "COMMENT ON COLUMN {table}.{column} IS {comment}"
                    ).format(
                        table=table_ident,
                        column=sql.Identifier(field_name),
                        comment=sql.Literal(field_info.description),
                    )
                    conn.execute(comment_query)

            logger.info(
                f"Successfully verified/created table: {schema_name}.{actual_table_name}"
            )

    def get_table_schema_description(
        self,
        table_name: str,
        schema_name: str = "duckdb",
    ) -> str:
        """_summary_

        :param table_name: _description_
        :param schema_name: _description_, defaults to "duckdb"
        :return: _description_
        """

        # get table schema and column descriptions
        query = sql.SQL("""
            SELECT 
                cols.column_name, 
                cols.data_type, 
                cols.is_nullable,
                (
                    SELECT pg_catalog.col_description(c.oid, cols.ordinal_position::int)
                    FROM pg_catalog.pg_class c
                    WHERE c.relname = cols.table_name
                    AND c.relnamespace = (SELECT oid FROM pg_catalog.pg_namespace WHERE nspname = cols.table_schema)
                ) as column_comment
            FROM information_schema.columns cols
            WHERE cols.table_schema = {schema}
            AND cols.table_name = {table}
            ORDER BY cols.ordinal_position;
        """).format(schema=sql.Literal(schema_name), table=sql.Literal(table_name))

        with self.pool.connection() as conn:
            conn.row_factory = dict_row
            rows = conn.execute(query).fetchall()

        if not rows:
            return f"Table {schema_name}.{table_name} not found."

        # build markdown string
        lines = [f"TABLE NAME:\n{schema_name}.{table_name}\n", "TABLE COLUMNS:"]
        for r in rows:
            null_str = " (Nullable)" if r["is_nullable"] == "YES" else " (NOT NULL)"
            comment = f" = {r['column_comment']}" if r["column_comment"] else ""
            lines.append(f"* {r['column_name']} ({r['data_type']}){null_str}{comment}")

        return "\n".join(lines)

    def insert_model(
        self,
        table_name: str,
        model: BaseModel,
        schema_name: str = "duckdb",
        conflict_target: str = "id",  # The column(s) that define a duplicate
    ):
        data = model.model_dump()
        columns = list(data.keys())
        values = [Jsonb(v) if isinstance(v, (list, dict)) else v for v in data.values()]

        # SQL with ON CONFLICT DO NOTHING (skips duplicates)
        query = sql.SQL("""
            INSERT INTO {table} ({fields}) 
            VALUES ({placeholders})
            ON CONFLICT ({target}) DO NOTHING
        """).format(
            table=sql.Identifier(schema_name, table_name),
            fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
            placeholders=sql.SQL(", ").join(sql.Placeholder() * len(columns)),
            target=sql.Identifier(conflict_target),
        )

        with self.pool.connection() as conn:
            conn.execute(query, values)

    def batch_insert_models(
        self,
        table_name: str,
        models: List[BaseModel],
        chunk_size: int = 5000,
        schema_name: str = "duckdb",
    ):
        """_summary_

        :param table_name: _description_
        :param models: _description_
        :param chunk_size: _description_, defaults to 5000
        :param schema_name: _description_, defaults to "duckdb"
        """
        if not models:
            return

        model_class = type(models[0])
        columns = list(model_class.model_fields.keys())

        target_table = sql.Identifier(schema_name, table_name)
        temp_table_name = f"tmp_{table_name}_{uuid.uuid4().hex[:8]}"
        temp_table = sql.Identifier(temp_table_name)

        with self.pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    # 1. Create the temp table
                    cur.execute(
                        sql.SQL("CREATE TEMP TABLE {temp} (LIKE {target})").format(
                            temp=temp_table,
                            target=target_table,
                        )
                    )

                    # bulk copy into temp
                    copy_query = sql.SQL("COPY {temp} ({fields}) FROM STDIN").format(
                        temp=temp_table,
                        fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
                    )

                    for i in range(0, len(models), chunk_size):
                        chunk = models[i : i + chunk_size]
                        with cur.copy(copy_query) as copy:
                            for m in chunk:
                                row = [
                                    Jsonb(v) if isinstance(v, (list, dict)) else v
                                    for v in m.model_dump(mode="python").values()
                                ]
                                copy.write_row(row)

                    # find rows in temp that aren't in target
                    merge_query = sql.SQL("""
                        INSERT INTO {target} ({fields})
                        SELECT {fields} FROM {temp}
                        EXCEPT
                        SELECT {fields} FROM {target}
                    """).format(
                        target=target_table,
                        temp=temp_table,
                        fields=sql.SQL(", ").join(map(sql.Identifier, columns)),
                    )
                    cur.execute(merge_query)

                    logger.info(
                        f"Deduplicated (Set-based) insert complete for {schema_name}.{table_name}"
                    )

    def query(
        self,
        query_sql: str,
        params: Any = None,
        peek: bool = False,
        schema_name: str = "duckdb",
    ) -> Any:
        """_summary_

        :param query_sql: _description_
        :param params: _description_, defaults to None
        :param peek: _description_, defaults to False
        :param schema_name: _description_, defaults to "duckdb"
        :return: _description_
        """
        with self.pool.connection() as conn:
            conn.row_factory = dict_row

            # set the search path so raw SQL doesn't need "schema." prefixes
            if schema_name:
                conn.execute(
                    sql.SQL("SET search_path TO {}, public").format(
                        sql.Identifier(schema_name)
                    )
                )

            cur = conn.execute(query_sql, params)

            # do your thing
            if not peek and cur.description is not None:
                return cur.fetchall()

            # peek
            return cur.rowcount

    def drop_table(
        self,
        table_name: str,
        schema_name: str = "duckdb",
        cascade: bool = True,
    ):
        """_summary_

        :param table_name: _description_
        :param schema_name: _description_, defaults to "duckdb"
        :param cascade: _description_, defaults to True
        """
        query = sql.SQL("DROP TABLE IF EXISTS {table} {cascade}").format(
            table=sql.Identifier(schema_name, table_name),
            cascade=sql.SQL("CASCADE") if cascade else sql.SQL(""),
        )

        try:
            with self.pool.connection() as conn:
                conn.execute(query)
                logger.info(
                    f"Successfully dropped table (if it existed): {schema_name}.{table_name}"
                )
        except Exception as e:
            logger.error(f"Failed to drop table {schema_name}.{table_name}: {e}")
            raise

    def close(self):
        """Shut down the pool when the app exits"""
        self.pool.close()
