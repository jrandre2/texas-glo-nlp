from __future__ import annotations

import sqlite3

from src import utils


def _seed_documents(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE documents (
            id INTEGER PRIMARY KEY,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            category TEXT,
            year INTEGER,
            quarter INTEGER
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO documents (id, filename, filepath, category, year, quarter)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (1, "a.pdf", "/tmp/a.pdf", "Harvey_5B_Performance", 2025, 4),
            (2, "b.pdf", "/tmp/b.pdf", "Harvey_57M_Performance", 2025, 3),
            (3, "c.pdf", "/tmp/c.pdf", "Mitigation_Performance", 2024, 4),
        ],
    )
    conn.commit()


def test_parse_usd_known_formats() -> None:
    assert utils.parse_usd("$1,234.56") == 1234.56
    assert utils.parse_usd("57.8M") == 57_800_000.0
    assert utils.parse_usd("$5.2 billion") == 5_200_000_000.0
    assert utils.parse_usd("$100k") == 100_000.0
    assert utils.parse_usd("1234") == 1234.0
    assert utils.parse_usd("$1,234.") == 1234.0
    assert utils.parse_usd("") is None
    assert utils.parse_usd("not a number") is None


def test_parse_filename_patterns() -> None:
    assert utils.parse_filename("drgr-2019-disasters-2025-q4.pdf") == {
        "disaster_code": "2019-disasters",
        "year": 2025,
        "quarter": 4,
    }
    assert utils.parse_filename("foo-q3-2022.pdf") == {
        "disaster_code": "foo",
        "year": 2022,
        "quarter": 3,
    }
    assert utils.parse_filename("no-quarter-info.pdf") == {
        "disaster_code": "unknown",
        "year": None,
        "quarter": None,
    }


def test_get_documents_by_category_accepts_list_tuple_generator() -> None:
    conn = sqlite3.connect(":memory:")
    _seed_documents(conn)

    rows_list = utils.get_documents_by_category(
        conn, ["Harvey_5B_Performance", "Harvey_57M_Performance"]
    )
    assert [r["id"] for r in rows_list] == [2, 1]

    rows_tuple = utils.get_documents_by_category(
        conn, ("Harvey_5B_Performance", "Harvey_57M_Performance")
    )
    assert [r["id"] for r in rows_tuple] == [2, 1]

    rows_gen = utils.get_documents_by_category(
        conn, (c for c in ["Harvey_5B_Performance", "Harvey_57M_Performance"])
    )
    assert [r["id"] for r in rows_gen] == [2, 1]


def test_get_documents_by_category_empty_iterable_returns_empty() -> None:
    conn = sqlite3.connect(":memory:")
    _seed_documents(conn)
    assert utils.get_documents_by_category(conn, []) == []
