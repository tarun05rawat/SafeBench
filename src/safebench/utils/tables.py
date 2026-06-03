from __future__ import annotations

from typing import Any


def markdown_table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return "_No rows available._"
    columns = columns or list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])

