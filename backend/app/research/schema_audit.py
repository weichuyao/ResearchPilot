"""Read-only compatibility audit for the Harness relational schema."""

from __future__ import annotations

from sqlalchemy import inspect
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import SQLModel


def _inspect_schema(connection) -> dict:
    inspector = inspect(connection)
    expected = {
        name: table for name, table in SQLModel.metadata.tables.items()
        if name.startswith("research_")
    }
    existing_names = set(inspector.get_table_names())
    missing_tables = sorted(set(expected) - existing_names)
    missing_columns: dict[str, list[str]] = {}
    type_warnings: dict[str, list[str]] = {}
    for name, table in expected.items():
        if name not in existing_names:
            continue
        actual = {column["name"]: column for column in inspector.get_columns(name)}
        missing = sorted(set(table.columns.keys()) - set(actual))
        if missing:
            missing_columns[name] = missing
        warnings = []
        for column in table.columns:
            if column.name not in actual:
                continue
            expected_type = column.type.compile(connection.dialect).lower()
            actual_type = str(actual[column.name]["type"]).lower()
            # Dialects render equivalent names differently (varchar vs character varying),
            # so only flag bounded string columns whose real capacity is too small.
            expected_length = getattr(column.type, "length", None)
            actual_length = getattr(actual[column.name]["type"], "length", None)
            if expected_length and actual_length and actual_length < expected_length:
                warnings.append(
                    f"{column.name}: capacity {actual_length} < expected {expected_length} "
                    f"({actual_type} vs {expected_type})"
                )
        if warnings:
            type_warnings[name] = warnings
    return {
        "dialect": connection.dialect.name,
        "expected_table_count": len(expected),
        "existing_table_count": len(set(expected) & existing_names),
        "missing_tables": missing_tables,
        "missing_columns": missing_columns,
        "type_warnings": type_warnings,
        "compatible": not missing_tables and not missing_columns and not type_warnings,
    }


async def inspect_research_schema(session: AsyncSession) -> dict:
    connection = await session.connection()
    return await connection.run_sync(_inspect_schema)


__all__ = ["inspect_research_schema"]
