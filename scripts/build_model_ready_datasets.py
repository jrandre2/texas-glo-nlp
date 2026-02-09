#!/usr/bin/env python3
"""
Build model-ready datasets for statistical modeling / EDA.

Reads from: data/glo_reports.db
Writes to:  outputs/model_ready/

Outputs are intentionally small(ish), tidy CSVs that can be committed to GitHub
and shared with non-technical stakeholders (alongside TEAM_PORTAL.html).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = ROOT / "data" / "glo_reports.db"
DEFAULT_OUT_DIR = ROOT / "outputs" / "model_ready"
TX_COUNTY_FIPS_CSV = ROOT / "data" / "reference" / "tx_county_fips.csv"


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_sql(con: sqlite3.Connection, query: str, params: Tuple[Any, ...] = ()) -> pd.DataFrame:
    return pd.read_sql_query(query, con, params=params)


def _cleanup_macos_artifacts(root: Path) -> int:
    """
    Remove macOS AppleDouble / Finder artifacts that confuse non-technical users.

    Returns: number of files removed
    """
    removed = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name == ".DS_Store" or name.startswith("._"):
                try:
                    Path(dirpath, name).unlink()
                    removed += 1
                except OSError:
                    continue
    return removed


def _normalize_key(key: str) -> str:
    k = key.strip()
    if k.endswith(":"):
        k = k[:-1]
    k = k.strip().lower()
    k = re.sub(r"[/\\-]+", " ", k)
    k = re.sub(r"[^a-z0-9]+", "_", k)
    k = re.sub(r"_+", "_", k).strip("_")
    return k


def _clean_text(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val)
    s = " ".join(s.split())
    s = s.strip()
    return s or None


def _parse_json_table(table_data: str) -> Optional[Any]:
    try:
        return json.loads(table_data)
    except Exception:
        return None


def _is_grid(table_obj: Any) -> bool:
    return isinstance(table_obj, list) and bool(table_obj) and isinstance(table_obj[0], list)


def _grid_to_text(table_obj: Any) -> str:
    if not isinstance(table_obj, list):
        return ""
    parts: List[str] = []
    if _is_grid(table_obj):
        for row in table_obj:
            for cell in row:
                c = _clean_text(cell)
                if c:
                    parts.append(c)
    else:
        for cell in table_obj:
            c = _clean_text(cell)
            if c:
                parts.append(c)
    return "\n".join(parts)


def _parse_kv_grid(table_grid: List[List[Any]]) -> Dict[str, Optional[str]]:
    """
    Parse DRGR key/value grids where rows alternate between labels and values.

    Example:
      ['Activity Type:', None, 'Activity Status:']
      ['Administration', None, 'Under Way']
      ['Project Number:', None, 'Project Title:']
      ['0001', None, 'ADMINISTRATION']
    """
    record: Dict[str, Optional[str]] = {}
    n_rows = len(table_grid)
    for i in range(0, n_rows - 1, 2):
        keys = table_grid[i]
        vals = table_grid[i + 1]
        for j, key_cell in enumerate(keys):
            key = _clean_text(key_cell)
            if not key or not key.endswith(":"):
                continue
            norm = _normalize_key(key)
            val = vals[j] if j < len(vals) else None
            record[norm] = _clean_text(val)
    return record


_STATUS_ORDER = ["Completed", "Under Way", "Cancelled", "Not Started"]


def _normalize_status(status: Optional[str]) -> Optional[str]:
    if not status:
        return None
    s = " ".join(status.split()).strip()
    s_low = s.lower()
    if "complete" in s_low:
        return "Completed"
    if "under way" in s_low or "underway" in s_low:
        return "Under Way"
    if "cancel" in s_low:
        return "Cancelled"
    if "not started" in s_low:
        return "Not Started"
    return s


def _extract_activity_code_title(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract Grantee Activity Number + Activity Title from page text.

    Handles:
      "Grantee Activity Number: GLO-Admin Activity Title: GLO"
      "Grantee Activity Number: Activity Title:\nCODE TITLE..."
    """
    if not text:
        return None, None

    def looks_like_currency(token: str) -> bool:
        return bool(re.fullmatch(r"\$?[\d,]+(?:\.\d{2})?", token or ""))

    def plausible_code(token: Optional[str]) -> bool:
        if not token:
            return False
        t = _clean_text(token) or ""
        if not t or len(t) < 3 or len(t) > 120:
            return False
        if t.endswith(":"):
            return False
        low = t.lower()
        if low in {"projected", "overall", "activity", "na", "n/a"}:
            return False
        if looks_like_currency(t):
            return False
        if re.fullmatch(r"[\d,]+(?:\.\d+)?", t):
            return False
        return bool(re.search(r"\d", t)) or low.startswith("glo-")

    # Same-line format
    m = re.search(
        r"Grantee Activity Number:\s*([^\n]+?)\s+Activity Title:\s*([^\n]+)",
        text,
        re.IGNORECASE,
    )
    if m:
        code = _clean_text(m.group(1))
        title = _clean_text(m.group(2))
        if plausible_code(code):
            return code, title
        return None, title

    # Split-line format
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for idx, ln in enumerate(lines):
        if "grantee activity number" in ln.lower():
            # if values on same line
            m2 = re.search(r"Grantee Activity Number:\s*(\S+)", ln, re.IGNORECASE)
            if m2 and "activity title" not in ln.lower():
                code = _clean_text(m2.group(1))
                if not plausible_code(code):
                    code = None
                title = None
                # try next line for title
                if idx + 1 < len(lines) and "activity title" in lines[idx + 1].lower():
                    m3 = re.search(r"Activity Title:\s*(.+)$", lines[idx + 1], re.IGNORECASE)
                    if m3:
                        title = _clean_text(m3.group(1))
                return code, title

            # label-only line -> next line has code + title
            for j in range(idx + 1, min(len(lines), idx + 8)):
                nxt = lines[j]
                if nxt.endswith(":"):
                    continue
                parts = nxt.split()
                if not parts:
                    continue
                code_token = _clean_text(parts[0].rstrip(",:;"))
                if not plausible_code(code_token):
                    continue
                title = _clean_text(" ".join(parts[1:])) if len(parts) > 1 else None
                return code_token, title
    return None, None


def _parse_activity_category_and_status(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Older DRGR formats sometimes label "Activity Category" instead of "Activity Type".
    The value line often looks like: "Planning Under Way".
    """
    if not text:
        return None, None

    m = re.search(
        r"Activity Category:\s*Activity Status:\s*\n?([^\n]+)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None, None
    val = _clean_text(m.group(1)) or ""
    status = None
    category = val
    for s in _STATUS_ORDER:
        if val.lower().endswith(s.lower()):
            status = s
            category = val[: -len(s)].strip()
            break
    return _clean_text(category), _normalize_status(status)


def _extract_first(text: str, pattern: str, flags: int = re.IGNORECASE) -> Optional[str]:
    m = re.search(pattern, text or "", flags)
    return _clean_text(m.group(1)) if m else None


def _parse_activity_from_textblock(text: str) -> Dict[str, Optional[str]]:
    """
    Parse key fields from a text block that contains DRGR activity page content.
    """
    out: Dict[str, Optional[str]] = {}
    if not text:
        return out

    # Grantee activity code/title
    code, title = _extract_activity_code_title(text)
    out["grantee_activity_number"] = code
    out["activity_title"] = title

    # Activity category + status (older format)
    cat, status = _parse_activity_category_and_status(text)
    if cat:
        out["activity_type"] = cat
    if status:
        out["activity_status"] = status

    # Project number/title (multiple formats)
    out["project_number"] = _extract_first(text, r"Project Number:\s*([^\n]+)")
    out["project_title"] = _extract_first(text, r"Project Title:\s*([^\n]+)")

    if not out["project_number"] or not out["project_title"]:
        m = re.search(r"Project\s*#\s*/\s*Project Title:\s*([0-9A-Za-z_-]+)\s*/\s*([^\n]+)", text, re.IGNORECASE)
        if m:
            out["project_number"] = out["project_number"] or _clean_text(m.group(1))
            out["project_title"] = out["project_title"] or _clean_text(m.group(2))

    # Common fields
    out["responsible_organization"] = _extract_first(text, r"Responsible Organization:\s*([^\n]+)")
    out["benefit_type"] = _extract_first(text, r"Benefit Type:\s*([^\n]+)")
    out["national_objective"] = _extract_first(text, r"National Objective:\s*([^\n]+)")
    out["projected_start_date"] = _extract_first(text, r"Projected Start Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})")
    out["projected_end_date"] = _extract_first(text, r"Projected End Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})")
    out["completed_activity_actual_end_date"] = _extract_first(
        text,
        r"Completed Activity Actual End Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})",
    )

    return out


def _extract_project_number(text: str) -> Optional[str]:
    if not text:
        return None
    # Common DRGR format
    m = re.search(r"Project Number:\s*\n?\s*([0-9A-Za-z_-]+)", text, re.IGNORECASE)
    if m:
        return _clean_text(m.group(1))
    # Alternate “0001 / TITLE” header
    m = re.search(r"^\s*(\d{4})\s*/\s*([^\n]+)", text, re.MULTILINE)
    if m:
        return _clean_text(m.group(1))
    return None


def _extract_project_title(text: str) -> Optional[str]:
    if not text:
        return None
    m = re.search(r"Project Title:\s*\n?\s*([^\n]+)", text, re.IGNORECASE)
    if m:
        return _clean_text(m.group(1))
    m = re.search(r"^\s*(\d{4})\s*/\s*([^\n]+)", text, re.MULTILINE)
    if m:
        return _clean_text(m.group(2))
    return None


def _extract_grantee_activity_number(text: str) -> Optional[str]:
    """
    Extract the DRGR "Grantee Activity Number" where present.

    Note: some PDFs do not place the value on the same line as the label.
    This tries multiple fallbacks.
    """
    code, _ = _extract_activity_code_title(text or "")
    if code:
        return code
    # Common coded IDs (e.g., 18-417-000_MI_Admin_Austin)
    m = re.search(r"(^|\n)\s*(\d{2}-\d{3}-\d{3}[^\n]*)", text or "")
    if m:
        return _clean_text(m.group(2))
    # Numeric-coded IDs (e.g., 72090001-Admin)
    m = re.search(r"(^|\n)\s*(\d{6,}[A-Za-z0-9_-]*-[A-Za-z][A-Za-z0-9_-]*)\s*(\n|$)", text or "")
    if m:
        return _clean_text(m.group(2))
    # GLO-style IDs (e.g., GLO-Admin)
    m = re.search(r"(^|\n)\s*(GLO-[A-Za-z0-9_-]+)", text or "", re.IGNORECASE)
    if m:
        return _clean_text(m.group(2))
    return None


def _extract_activity_status_and_type(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract activity status and a best-effort activity type from page text.

    Strategy: find a line that equals a known status and treat the nearest
    preceding non-label, non-currency line as the type.
    """
    if not text:
        return None, None

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    status_idx = None
    status_val = None
    for i, ln in enumerate(lines):
        norm = _normalize_status(ln)
        if norm in {"Completed", "Under Way", "Cancelled", "Not Started"} and ln == norm:
            status_idx = i
            status_val = norm
            break
    if status_idx is None:
        return None, None

    def is_currency(s: str) -> bool:
        return bool(re.fullmatch(r"\$?\s*[\d,]+\.\d{2}\s*", s))

    activity_type = None
    for j in range(status_idx - 1, -1, -1):
        candidate = lines[j]
        if candidate.endswith(":"):
            continue
        if is_currency(candidate):
            continue
        if candidate.lower() in {"n/a", "na"}:
            continue
        activity_type = candidate
        break

    return status_val, _clean_text(activity_type)


_BENEFICIARY_ROW_RE = re.compile(
    r"Low/Mod\s+"
    r"(?P<this_period>\d+(?:\.\d+)?)\s+"
    r"(?P<low>\d+/\d+)\s+"
    r"(?P<mod>\d+/\d+)\s+"
    r"(?P<total>\d+/\d+)\s+"
    r"#\s*(?:of\s+)?(?P<measure>[A-Za-z][A-Za-z ]+?)\s+"
    r"(?P<post1>\d+(?:\.\d+)?)\s+"
    r"(?P<post2>\d+(?:\.\d+)?)\s+"
    r"(?P<post3>\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def _parse_actual_expected(token: str) -> Tuple[Optional[int], Optional[int]]:
    if not token or "/" not in token:
        return None, None
    a, b = token.split("/", 1)
    try:
        return int(a), int(b)
    except ValueError:
        return None, None


def _canonicalize_beneficiary_measure(raw_measure: str) -> Optional[str]:
    """
    Map messy DRGR measure labels to a small canonical set.
    """
    if not raw_measure:
        return None
    m = _normalize_key(raw_measure)
    if "person" in m:
        return "persons"
    if ("household" in m and "owner" in m) or m in {"owner", "owners"}:
        return "owner_households"
    if ("household" in m and "renter" in m) or m in {"renter", "renters"}:
        return "renter_households"
    if "household" in m:
        return "households"
    if "job" in m and ("retain" in m or "retained" in m):
        return "jobs_retained"
    if "job" in m and ("create" in m or "created" in m or "permanent" in m):
        return "jobs_created"
    if ("housing" in m and "unit" in m) or "singlefamily_unit" in m or "multifamily_unit" in m:
        return "housing_units"
    return None


def _extract_lmi_percent(text: str) -> Optional[float]:
    m = re.search(r"LMI%:\s*([0-9]+(?:\.[0-9]+)?)", text or "", re.IGNORECASE)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


_BENEFICIARY_ACTUAL_EXPECTED_RE = re.compile(
    r"(?P<actual>\d[\d,]*)\s*/\s*(?P<expected>\d[\d,]*)\s+"
    r"#\s*(?:of\s+)?(?P<measure>[A-Za-z][A-Za-z /\-]+)"
    r"(?:\s+(?P<this_period>\d[\d,]*))?\s*$",
    re.IGNORECASE,
)

_BENEFICIARY_PROPOSED_RE = re.compile(
    r"#\s*(?:of\s+)?(?P<measure>[A-Za-z][A-Za-z /\-]+?)\s+"
    r"(?P<total>\d[\d,]*)"
    r"(?:\s+(?P<aux1>\d[\d,]*))?"
    r"(?:\s+(?P<aux2>\d+(?:\.\d+)?%?))?(?:\s|$)",
    re.IGNORECASE,
)


def _parse_number_token(token: Optional[str]) -> Optional[float]:
    if not token:
        return None
    s = token.strip().replace(",", "")
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _row_record(
    raw_measure: Optional[str],
    canonical: Optional[str],
    this_period: Optional[float] = None,
    low_actual: Optional[int] = None,
    low_expected: Optional[int] = None,
    mod_actual: Optional[int] = None,
    mod_expected: Optional[int] = None,
    total_actual: Optional[int] = None,
    total_expected: Optional[int] = None,
    post1: Optional[float] = None,
    post2: Optional[float] = None,
    post3: Optional[float] = None,
    lmi_percent: Optional[float] = None,
) -> Dict[str, Any]:
    return {
        "measure_raw": raw_measure,
        "measure": canonical or _normalize_key(raw_measure or ""),
        "canonical_measure": canonical,
        "this_period": this_period,
        "low_actual": low_actual,
        "low_expected": low_expected,
        "mod_actual": mod_actual,
        "mod_expected": mod_expected,
        "total_actual": total_actual,
        "total_expected": total_expected,
        "post1": post1,
        "post2": post2,
        "post3": post3,
        "lmi_percent": lmi_percent,
    }


def _parse_beneficiary_rows(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not text:
        return rows

    lmi_percent = _extract_lmi_percent(text)
    seen: set[Tuple[str, Any, Any, Any]] = set()

    # Legacy low/mod row format.
    for m in _BENEFICIARY_ROW_RE.finditer(text):
        low_a, low_e = _parse_actual_expected(m.group("low"))
        mod_a, mod_e = _parse_actual_expected(m.group("mod"))
        tot_a, tot_e = _parse_actual_expected(m.group("total"))
        this_period = _parse_number_token(m.group("this_period"))
        post1 = _parse_number_token(m.group("post1"))
        post2 = _parse_number_token(m.group("post2"))
        post3 = _parse_number_token(m.group("post3"))
        raw_measure = _clean_text(m.group("measure"))
        canonical = _canonicalize_beneficiary_measure(raw_measure or "")
        if not canonical:
            continue
        key = (canonical, this_period, tot_a, tot_e)
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            _row_record(
                raw_measure=raw_measure,
                canonical=canonical,
                this_period=this_period,
                low_actual=low_a,
                low_expected=low_e,
                mod_actual=mod_a,
                mod_expected=mod_e,
                total_actual=tot_a,
                total_expected=tot_e,
                post1=post1,
                post2=post2,
                post3=post3,
                lmi_percent=lmi_percent,
            )
        )

    # Additional formats are easier to parse line-by-line.
    lines = [_clean_text(ln) for ln in text.splitlines()]
    for line in [ln for ln in lines if ln]:
        if "#" not in line:
            continue
        low_line = line.lower()
        if "public facilit" in low_line or "linear feet" in low_line or "properties" in low_line:
            continue

        # Actual/expected accomplishments format: "280/281 # of Housing Units 0"
        am = _BENEFICIARY_ACTUAL_EXPECTED_RE.search(line)
        if am:
            raw_measure = _clean_text(am.group("measure"))
            canonical = _canonicalize_beneficiary_measure(raw_measure or "")
            if canonical:
                total_actual = _parse_number_token(am.group("actual"))
                total_expected = _parse_number_token(am.group("expected"))
                this_period = _parse_number_token(am.group("this_period"))
                key = (canonical, this_period, total_actual, total_expected)
                if key not in seen:
                    seen.add(key)
                    rows.append(
                        _row_record(
                            raw_measure=raw_measure,
                            canonical=canonical,
                            this_period=this_period,
                            total_actual=int(total_actual) if total_actual is not None else None,
                            total_expected=int(total_expected) if total_expected is not None else None,
                            lmi_percent=lmi_percent,
                        )
                    )
            continue

        # Proposed beneficiaries/accomplishments format:
        # "# of Households 1415 1348 95.27%" or "# Renter Households 1604 1604 100.00%"
        pm = _BENEFICIARY_PROPOSED_RE.search(line)
        if pm:
            raw_measure = _clean_text(pm.group("measure"))
            canonical = _canonicalize_beneficiary_measure(raw_measure or "")
            if canonical:
                total_expected = _parse_number_token(pm.group("total"))
                aux1 = _parse_number_token(pm.group("aux1"))
                # If both total and aux1 are present, treat aux1 as low/mod count (not total_actual).
                # We keep total_expected for SEM and leave actual as unknown unless we have x/y format.
                low_actual = int(aux1) if (aux1 is not None and "household" in (canonical or "")) else None
                key = (canonical, None, None, total_expected)
                if key not in seen:
                    seen.add(key)
                    rows.append(
                        _row_record(
                            raw_measure=raw_measure,
                            canonical=canonical,
                            total_actual=None,
                            total_expected=int(total_expected) if total_expected is not None else None,
                            low_actual=low_actual,
                            lmi_percent=lmi_percent,
                        )
                    )

    return rows


def _activity_type_group(raw_type: Optional[str]) -> str:
    t = (raw_type or "").lower()
    if not t:
        return "Unknown"
    if "admin" in t:
        return "Administration"
    if "planning" in t or t == "planning":
        return "Planning"
    if "buyout" in t or "acquisition" in t:
        return "Acquisition/Buyout"
    if "rental" in t or "homeowner" in t or "housing" in t:
        return "Housing"
    if any(k in t for k in ["drainage", "water", "sewer", "street", "infrastructure", "public", "facility", "improvement"]):
        return "Infrastructure"
    return "Other"


_SEM_MONEY_TOKEN_RE = re.compile(
    r"(?:US\$|\$)\s*\d[\d,]*(?:\.\d+)?(?:\s*(?:million|billion|m|b|k))?"
    r"|"
    r"\d[\d,]*(?:\.\d+)?\s*(?:million|billion|m|b)\b",
    re.IGNORECASE,
)
_SEM_STAFF_COUNT_RE = re.compile(
    r"(?<![\$])(?P<count>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<label>fte|f\.t\.e\.|full[\-\s]*time equivalent(?:s)?|headcount|staff(?:ers)?|employees?|personnel|positions?)\b",
    re.IGNORECASE,
)
_SEM_DEATH_RE = re.compile(
    r"(?P<count>\d[\d,]*)\s+(?:direct\s+|indirect\s+)?(?:deaths?|fatalities|fatality|casualties)\b",
    re.IGNORECASE,
)
_SEM_KILLING_RE = re.compile(
    r"killing\s+(?P<count>\d[\d,]*)\s+(?:people|persons|residents|texans|individuals)\b",
    re.IGNORECASE,
)
_SEM_AFFECTED_AFTER_RE = re.compile(
    r"(?P<count>\d[\d,]*)\s+(?P<unit>people|persons|residents|households|families)\b[^\.]{0,80}\b"
    r"(?:affected|impacted|displaced|evacuated|assisted|benefited|rescued)\b",
    re.IGNORECASE,
)
_SEM_AFFECTED_BEFORE_RE = re.compile(
    r"(?:affected|impacted|displaced|evacuated|assisted|benefited|rescued)\s+(?:an\s+estimated\s+)?"
    r"(?P<count>\d[\d,]*)\s+(?P<unit>people|persons|residents|households|families)\b",
    re.IGNORECASE,
)
_SEM_TABLE_KEYWORD_RE = re.compile(
    r"payroll|salary|wage|staff|fte|headcount|personnel|death|fatal|economic loss|property damage|unmet need|"
    r"households|owner households|renter households|jobs|housing units|affected|impacted|displaced",
    re.IGNORECASE,
)
_SEM_DATE_FMT = ("%m/%d/%Y", "%m/%d/%y")


def _iter_sem_chunks(text: str) -> Iterable[str]:
    for raw_line in (text or "").splitlines():
        line = _clean_text(raw_line)
        if not line:
            continue
        if len(line) <= 260:
            yield line
            continue
        for part in re.split(r"(?<=[\.;])\s+", line):
            p = _clean_text(part)
            if p:
                yield p


def _parse_sem_usd_token(token: str, context_line: str) -> Optional[float]:
    t = token.strip().replace(",", "")
    t = t.replace("US$", "$").replace(" ", "")
    if not t:
        return None
    if t.startswith("$"):
        body = t[1:]
        has_currency_hint = True
    else:
        body = t
        has_currency_hint = False

    m = re.fullmatch(r"(?P<num>\d+(?:\.\d+)?)(?P<suf>million|billion|m|b|k)?", body, re.IGNORECASE)
    if not m:
        return None
    n = float(m.group("num"))
    suf = (m.group("suf") or "").lower()
    mult = 1.0
    if suf in {"billion", "b"}:
        mult = 1_000_000_000.0
    elif suf in {"million", "m"}:
        mult = 1_000_000.0
    elif suf == "k":
        mult = 1_000.0

    if not has_currency_hint and not suf:
        if "dollar" not in (context_line or "").lower() and "usd" not in (context_line or "").lower():
            return None
    return n * mult


def _parse_int_token(token: Optional[str]) -> Optional[int]:
    v = _parse_number_token(token)
    if v is None:
        return None
    try:
        return int(round(float(v)))
    except Exception:
        return None


def _sem_metric_row(
    metric: str,
    value: float,
    unit: str,
    confidence: float,
    method: str,
    snippet: str,
) -> Dict[str, Any]:
    return {
        "metric": metric,
        "value": float(value),
        "unit": unit,
        "confidence": float(max(0.0, min(1.0, confidence))),
        "method": method,
        "snippet": (snippet or "")[:240],
    }


def _extract_sem_signals_from_text(text: str, source_type: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, float, str]] = set()
    source_penalty = 0.05 if source_type == "table_json" else 0.0

    for line in _iter_sem_chunks(text):
        low = line.lower()

        # Administrative payroll-like monetary mentions
        if any(k in low for k in ["payroll", "salary", "wage", "personnel cost", "staffing cost", "staff costs"]):
            for m in _SEM_MONEY_TOKEN_RE.finditer(line):
                usd = _parse_sem_usd_token(m.group(0), line)
                if usd is None:
                    continue
                conf = 0.9 if "payroll" in low else 0.78
                row = _sem_metric_row(
                    metric="admin_payroll_usd",
                    value=usd,
                    unit="usd",
                    confidence=conf - source_penalty,
                    method=f"{source_type}:keyword_money",
                    snippet=line,
                )
                key = (row["metric"], row["value"], row["snippet"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

        # Administrative staffing counts
        if any(k in low for k in ["fte", "headcount", "staff", "personnel", "employee", "position"]):
            for m in _SEM_STAFF_COUNT_RE.finditer(line):
                span = line[m.start() : m.end()]
                if "%" in span or "percent" in span.lower():
                    continue
                count = _parse_number_token(m.group("count"))
                if count is None or count <= 0:
                    continue
                label = (m.group("label") or "").lower()
                conf = 0.92 if ("fte" in label or "headcount" in label) else 0.74
                row = _sem_metric_row(
                    metric="admin_staff_count",
                    value=count,
                    unit="count",
                    confidence=conf - source_penalty,
                    method=f"{source_type}:staff_pattern",
                    snippet=line,
                )
                key = (row["metric"], row["value"], row["snippet"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

        # Death/casualty counts
        for rgx, method_name in [(_SEM_DEATH_RE, "death_pattern"), (_SEM_KILLING_RE, "killing_pattern")]:
            for m in rgx.finditer(line):
                count = _parse_number_token(m.group("count"))
                if count is None or count <= 0:
                    continue
                row = _sem_metric_row(
                    metric="severity_deaths_count",
                    value=count,
                    unit="count",
                    confidence=0.9 - source_penalty,
                    method=f"{source_type}:{method_name}",
                    snippet=line,
                )
                key = (row["metric"], row["value"], row["snippet"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

        # Affected population counts (persons/households)
        for rgx, method_name in [(_SEM_AFFECTED_AFTER_RE, "affected_after"), (_SEM_AFFECTED_BEFORE_RE, "affected_before")]:
            for m in rgx.finditer(line):
                count = _parse_number_token(m.group("count"))
                unit = (m.group("unit") or "").lower()
                if count is None or count <= 0:
                    continue
                metric = "affected_population_persons" if unit in {"people", "persons", "residents"} else "affected_population_households"
                row = _sem_metric_row(
                    metric=metric,
                    value=count,
                    unit="count",
                    confidence=0.76 - source_penalty,
                    method=f"{source_type}:{method_name}",
                    snippet=line,
                )
                key = (row["metric"], row["value"], row["snippet"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

        # Monetary severity constructs.
        severity_metric = None
        if "unmet need" in low:
            severity_metric = "severity_unmet_need_usd"
        elif "property damage" in low:
            severity_metric = "severity_property_damage_usd"
        elif "economic loss" in low or "economic losses" in low or ("loss" in low and "damage" in low):
            severity_metric = "severity_economic_loss_usd"
        if severity_metric:
            for m in _SEM_MONEY_TOKEN_RE.finditer(line):
                usd = _parse_sem_usd_token(m.group(0), line)
                if usd is None:
                    continue
                row = _sem_metric_row(
                    metric=severity_metric,
                    value=usd,
                    unit="usd",
                    confidence=0.84 - source_penalty,
                    method=f"{source_type}:severity_money",
                    snippet=line,
                )
                key = (row["metric"], row["value"], row["snippet"])
                if key not in seen:
                    seen.add(key)
                    rows.append(row)

    return rows


def _table_data_to_text(table_data: str) -> str:
    obj = _parse_json_table(table_data or "")
    if obj is None:
        return _clean_text(table_data) or ""
    if _is_grid(obj):
        lines: List[str] = []
        for row in obj:
            cells = [_clean_text(c) for c in row]
            clean_cells = [c for c in cells if c]
            if clean_cells:
                lines.append(" | ".join(clean_cells))
        return "\n".join(lines)
    flat = _grid_to_text(obj)
    return flat or ""


def _parse_date_token(value: Optional[str]) -> Optional[datetime]:
    v = _clean_text(value)
    if not v:
        return None
    for fmt in _SEM_DATE_FMT:
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def _duration_quarters(start: Optional[datetime], end: Optional[datetime]) -> Optional[float]:
    if start is None or end is None:
        return None
    if end < start:
        return None
    days = (end - start).days
    return float(int(days / 91.25) + 1)


@dataclass(frozen=True)
class CountyFipsIndex:
    by_token: Dict[str, Tuple[str, str]]  # token -> (county_name, fips3)
    by_fips3: Dict[str, str]  # fips3 -> county_name

    @staticmethod
    def load(path: Path) -> "CountyFipsIndex":
        df = pd.read_csv(path)
        by_token: Dict[str, Tuple[str, str]] = {}
        by_fips3: Dict[str, str] = {}
        for _, row in df.iterrows():
            name = str(row["county"]).strip()
            fips3 = str(row["fips"]).strip().zfill(3)
            token = CountyFipsIndex._tokenize(name)
            by_token[token] = (name, fips3)
            by_fips3[fips3] = name
        return CountyFipsIndex(by_token=by_token, by_fips3=by_fips3)

    @staticmethod
    def _tokenize(value: str) -> str:
        v = value.lower()
        v = v.replace("county", "")
        v = re.sub(r"[^a-z]+", "", v)
        return v

    def match(self, value: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        if not value:
            return None, None
        token = self._tokenize(str(value))
        hit = self.by_token.get(token)
        if not hit:
            return None, None
        return hit[0], hit[1]


def _quality_report(
    row_counts: Dict[str, int],
    documents: pd.DataFrame,
    panel_disaster_quarter: pd.DataFrame,
    activities_unique: pd.DataFrame,
    money_by_quarter: pd.DataFrame,
    max_quarter_drop_pct: float,
) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    failures: List[str] = []

    required_non_empty = {
        "panel_document": 1,
        "panel_disaster_quarter": 1,
        "activities": 1,
        "activities_unique": 1,
        "money_mentions_by_quarter": 1,
        "panel_state_quarter": 1,
        "panel_disaster_quarter_sem": 1,
        "panel_county_quarter_sem": 1,
        "panel_city_quarter_sem": 1,
        "panel_state_quarter_sem": 1,
        "sem_construct_signals": 1,
        "sem_coverage_report": 1,
        "sem_variable_dictionary": 1,
    }
    for key, min_rows in required_non_empty.items():
        got = int(row_counts.get(key, 0))
        ok = got >= int(min_rows)
        checks.append(
            {
                "type": "non_empty",
                "dataset": key,
                "min_rows": int(min_rows),
                "row_count": got,
                "ok": ok,
            }
        )
        if not ok:
            failures.append(f"{key}: expected at least {min_rows} rows, found {got}")

    def quarter_pairs(df: pd.DataFrame) -> List[Tuple[int, int]]:
        if df.empty or "year" not in df.columns or "quarter" not in df.columns:
            return []
        q = (
            df[["year", "quarter"]]
            .dropna()
            .astype({"year": int, "quarter": int})
            .drop_duplicates()
            .sort_values(["year", "quarter"])
        )
        return [(int(r.year), int(r.quarter)) for r in q.itertuples(index=False)]

    pairs = quarter_pairs(documents)
    if len(pairs) >= 2:
        latest_y, latest_q = pairs[-1]
        prev_y, prev_q = pairs[-2]

        def docs_count(y: int, q: int) -> float:
            rows = documents[(documents["year"] == y) & (documents["quarter"] == q)]
            return float(len(rows))

        def panel_rows(y: int, q: int) -> float:
            if panel_disaster_quarter.empty:
                return 0.0
            rows = panel_disaster_quarter[
                (panel_disaster_quarter["year"] == y) & (panel_disaster_quarter["quarter"] == q)
            ]
            return float(len(rows))

        def activity_count(y: int, q: int) -> float:
            if activities_unique.empty:
                return 0.0
            rows = activities_unique[(activities_unique["year"] == y) & (activities_unique["quarter"] == q)]
            return float(len(rows))

        def money_rows(y: int, q: int) -> float:
            if money_by_quarter.empty:
                return 0.0
            rows = money_by_quarter[(money_by_quarter["year"] == y) & (money_by_quarter["quarter"] == q)]
            return float(len(rows))

        delta_metrics = {
            "documents_latest_quarter_count": docs_count,
            "panel_disaster_quarter_row_count": panel_rows,
            "activities_unique_count": activity_count,
            "money_mentions_by_quarter_row_count": money_rows,
        }
        for name, fn in delta_metrics.items():
            prev_value = float(fn(prev_y, prev_q))
            latest_value = float(fn(latest_y, latest_q))
            drop_pct = 0.0
            if prev_value > 0:
                drop_pct = max(0.0, ((prev_value - latest_value) / prev_value) * 100.0)
            ok = True
            if prev_value > 0 and drop_pct > float(max_quarter_drop_pct):
                ok = False
            checks.append(
                {
                    "type": "prior_quarter_delta",
                    "metric": name,
                    "latest_quarter": f"Q{latest_q} {latest_y}",
                    "prev_quarter": f"Q{prev_q} {prev_y}",
                    "latest_value": latest_value,
                    "prev_value": prev_value,
                    "drop_pct": round(drop_pct, 2),
                    "max_drop_pct": float(max_quarter_drop_pct),
                    "ok": ok,
                }
            )
            if not ok:
                failures.append(
                    f"{name}: dropped {drop_pct:.2f}% from Q{prev_q} {prev_y} to Q{latest_q} {latest_y} "
                    f"(max allowed {max_quarter_drop_pct:.2f}%)"
                )
    else:
        checks.append(
            {
                "type": "prior_quarter_delta",
                "metric": "all",
                "ok": True,
                "note": "Not enough quarters available to run quarter-over-quarter checks.",
            }
        )

    return {
        "built_at": _now_iso(),
        "max_quarter_drop_pct": float(max_quarter_drop_pct),
        "checks": checks,
        "failures": failures,
        "ok": len(failures) == 0,
    }


def build_outputs(
    db_path: Path,
    out_dir: Path,
    allow_partial: bool = False,
    max_quarter_drop_pct: float = 80.0,
) -> Dict[str, Any]:
    _safe_mkdir(out_dir)
    _safe_mkdir(out_dir / "panels")
    _safe_mkdir(out_dir / "long")
    _safe_mkdir(out_dir / "meta")

    con = sqlite3.connect(str(db_path))

    # ---- documents (base index) ----
    documents = _read_sql(
        con,
        """
        SELECT
          id AS document_id,
          filename,
          category,
          disaster_code,
          year,
          quarter,
          page_count,
          file_size_bytes,
          processed_at
        FROM documents
        ORDER BY year, quarter, category, filename
        """,
    )
    documents["quarter_label"] = documents.apply(
        lambda r: f"Q{int(r['quarter'])} {int(r['year'])}" if pd.notna(r["year"]) and pd.notna(r["quarter"]) else None,
        axis=1,
    )

    # ---- money aggregates ----
    money_by_doc = _read_sql(
        con,
        """
        SELECT
          document_id,
          context_label,
          COUNT(*) AS n_mentions,
          SUM(amount_usd) AS sum_amount_usd,
          MAX(amount_usd) AS max_amount_usd
        FROM money_mentions
        GROUP BY document_id, context_label
        """,
    )

    money_doc_wide = (
        money_by_doc.pivot_table(
            index="document_id",
            columns="context_label",
            values=["n_mentions", "sum_amount_usd", "max_amount_usd"],
            aggfunc="first",
        )
        .sort_index(axis=1)
    )
    money_doc_wide.columns = [f"money_{metric}_{label}" for metric, label in money_doc_wide.columns]
    money_doc_wide = money_doc_wide.reset_index()

    # ---- entities by document + by quarter ----
    entity_by_doc = _read_sql(
        con,
        """
        SELECT document_id, entity_type, COUNT(*) AS n_mentions
        FROM entities
        GROUP BY document_id, entity_type
        """,
    )
    entity_doc_wide = entity_by_doc.pivot_table(
        index="document_id",
        columns="entity_type",
        values="n_mentions",
        aggfunc="first",
    ).fillna(0)
    entity_doc_wide.columns = [f"entity_{c}" for c in entity_doc_wide.columns]
    entity_doc_wide = entity_doc_wide.reset_index()

    entity_by_quarter = _read_sql(
        con,
        """
        SELECT
          d.category,
          d.disaster_code,
          d.year,
          d.quarter,
          e.entity_type,
          COUNT(*) AS n_mentions
        FROM entities e
        JOIN documents d ON d.id = e.document_id
        GROUP BY d.category, d.disaster_code, d.year, d.quarter, e.entity_type
        ORDER BY d.category, d.year, d.quarter, e.entity_type
        """,
    )

    # ---- severity proxies (from extracted entities) ----
    severity_entities = _read_sql(
        con,
        """
        SELECT
          e.document_id,
          d.category,
          d.disaster_code,
          d.year,
          d.quarter,
          e.entity_type,
          e.entity_text
        FROM entities e
        JOIN documents d ON d.id = e.document_id
        WHERE e.entity_type IN ('RAINFALL', 'WIND_SPEED', 'FEMA_DECLARATION')
        """,
    )

    def parse_inches(s: str) -> Optional[float]:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*in\b", s or "", re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def parse_mph(s: str) -> Optional[float]:
        m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*mph\b", s or "", re.IGNORECASE)
        if not m:
            return None
        try:
            return float(m.group(1))
        except ValueError:
            return None

    def parse_fema_number(s: str) -> Optional[int]:
        m = re.search(r"(\d{4})", s or "")
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    severity_rows: List[pd.DataFrame] = []
    if not severity_entities.empty:
        rain = severity_entities[severity_entities["entity_type"] == "RAINFALL"].copy()
        if not rain.empty:
            rain["value"] = rain["entity_text"].map(parse_inches)
            severity_rows.append(
                rain.groupby(["category", "disaster_code", "year", "quarter"], dropna=False)
                .agg(n_mentions=("entity_text", "count"), max_value=("value", "max"))
                .reset_index()
                .assign(metric="rainfall_inches")
            )

        wind = severity_entities[severity_entities["entity_type"] == "WIND_SPEED"].copy()
        if not wind.empty:
            wind["value"] = wind["entity_text"].map(parse_mph)
            severity_rows.append(
                wind.groupby(["category", "disaster_code", "year", "quarter"], dropna=False)
                .agg(n_mentions=("entity_text", "count"), max_value=("value", "max"))
                .reset_index()
                .assign(metric="wind_speed_mph")
            )

    severity_proxies_by_quarter = (
        pd.concat(severity_rows, ignore_index=True) if severity_rows else pd.DataFrame(columns=["category", "disaster_code", "year", "quarter", "metric", "n_mentions", "max_value"])
    )

    # FEMA declarations (for external joining)
    fema_declarations_by_quarter = pd.DataFrame(
        columns=["category", "disaster_code", "year", "quarter", "fema_number", "n_mentions", "n_documents"]
    )
    if not severity_entities.empty:
        fema = severity_entities[severity_entities["entity_type"] == "FEMA_DECLARATION"].copy()
        if not fema.empty:
            fema["fema_number"] = fema["entity_text"].map(parse_fema_number)
            fema = fema.dropna(subset=["fema_number"])
            if not fema.empty:
                fema_declarations_by_quarter = (
                    fema.groupby(["category", "disaster_code", "year", "quarter", "fema_number"], dropna=False)
                    .agg(n_mentions=("entity_text", "count"), n_documents=("document_id", pd.Series.nunique))
                    .reset_index()
                    .sort_values(["category", "year", "quarter", "fema_number"])
                )

    # ---- topics by quarter ----
    topic_by_quarter = _read_sql(
        con,
        """
        SELECT
          ta.model_id,
          ta.topic_index,
          t.label AS topic_label,
          d.category,
          d.disaster_code,
          d.year,
          d.quarter,
          COUNT(*) AS n_chunks,
          COUNT(DISTINCT ta.document_id) AS n_documents
        FROM topic_assignments ta
        JOIN topics t
          ON t.model_id = ta.model_id
         AND t.topic_index = ta.topic_index
        JOIN documents d ON d.id = ta.document_id
        GROUP BY ta.model_id, ta.topic_index, t.label, d.category, d.disaster_code, d.year, d.quarter
        ORDER BY ta.model_id, ta.topic_index, d.category, d.year, d.quarter
        """,
    )

    # ---- keyword coverage by quarter (pages/documents) ----
    keywords = [
        "payroll",
        "headcount",
        "fte",
        "fatality",
        "death",
        "economic loss",
        "property damage",
        "unmet need",
    ]
    kw_rows: List[pd.DataFrame] = []
    for kw in keywords:
        like = f"%{kw}%"
        df = _read_sql(
            con,
            """
            SELECT
              ? AS keyword,
              d.category,
              d.disaster_code,
              d.year,
              d.quarter,
              COUNT(*) AS n_pages,
              COUNT(DISTINCT dt.document_id) AS n_documents
            FROM document_text dt
            JOIN documents d ON d.id = dt.document_id
            WHERE dt.text_content LIKE ? COLLATE NOCASE
            GROUP BY d.category, d.disaster_code, d.year, d.quarter
            """,
            params=(kw, like),
        )
        kw_rows.append(df)
    keyword_pages_by_quarter = pd.concat(kw_rows, ignore_index=True)

    # ---- activities (all disasters; raw-text based) ----
    county_index = CountyFipsIndex.load(TX_COUNTY_FIPS_CSV) if TX_COUNTY_FIPS_CSV.exists() else None
    cur = con.cursor()

    activities_rows: List[Dict[str, Any]] = []
    beneficiary_rows: List[Dict[str, Any]] = []
    sem_signal_rows: List[Dict[str, Any]] = []

    def city_ok(city: str) -> bool:
        c = city.strip()
        if not c or len(c) > 60:
            return False
        if ":" in c:
            return False
        bad = {
            "community development systems",
            "disaster recovery grant reporting system (drgr)",
            "drgr",
            "none",
        }
        return c.lower() not in bad

    for doc in documents.itertuples(index=False):
        doc_id = int(doc.document_id)
        doc_year = int(doc.year) if pd.notna(doc.year) else None
        doc_quarter = int(doc.quarter) if pd.notna(doc.quarter) else None

        # Fetch all pages (raw text) for grouping and SEM extraction.
        cur.execute(
            """
            SELECT page_number, raw_text_content
            FROM document_text
            WHERE document_id = ?
            ORDER BY page_number
            """,
            (doc_id,),
        )
        pages: List[Tuple[int, str]] = [(int(r[0]), r[1] or "") for r in cur.fetchall()]
        if not pages:
            continue

        header_idxs = [i for i, (_, txt) in enumerate(pages) if "Activity Status:" in (txt or "")]

        # Money stats by page for this document
        cur.execute(
            """
            SELECT page_number, context_label,
                   COUNT(*) AS n_mentions,
                   SUM(amount_usd) AS sum_amount_usd,
                   MAX(amount_usd) AS max_amount_usd
            FROM money_mentions
            WHERE document_id = ?
            GROUP BY page_number, context_label
            """,
            (doc_id,),
        )
        money_by_page: Dict[int, Dict[str, Dict[str, float]]] = {}
        for page_number, label, n, s, mx in cur.fetchall():
            page = int(page_number)
            d = money_by_page.setdefault(page, {})
            lbl = str(label)
            d[lbl] = {
                "n_mentions": float(n or 0),
                "sum_amount_usd": float(s or 0.0),
                "max_amount_usd": float(mx or 0.0),
            }

        # Geo hint scores by page (county + city + zip)
        page_county_any: Dict[int, Dict[str, float]] = {}
        page_county_direct: Dict[int, Dict[str, float]] = {}
        page_city: Dict[int, Dict[str, float]] = {}
        page_zip: Dict[int, Dict[str, float]] = {}

        if county_index is not None:
            cur.execute(
                """
                SELECT page_number, county, city, zip, confidence, method
                FROM location_mentions
                WHERE document_id = ?
                  AND method != 'table_header'
                """,
                (doc_id,),
            )
            for page_number, county, city, zip_code, conf, method in cur.fetchall():
                page = int(page_number)
                score = float(conf or 0.0)
                method_s = str(method or "")

                if county:
                    _name, fips3 = county_index.match(str(county))
                    if fips3:
                        page_county_any.setdefault(page, {})[fips3] = page_county_any.get(page, {}).get(fips3, 0.0) + score
                        if "derived_doc_county" not in method_s:
                            page_county_direct.setdefault(page, {})[fips3] = (
                                page_county_direct.get(page, {}).get(fips3, 0.0) + score
                            )

                if city:
                    city_clean = _clean_text(city)
                    if city_clean and city_ok(city_clean):
                        page_city.setdefault(page, {})[city_clean] = page_city.get(page, {}).get(city_clean, 0.0) + score

                if zip_code:
                    z = _clean_text(zip_code)
                    if z and re.fullmatch(r"\d{5}", z):
                        page_zip.setdefault(page, {})[z] = page_zip.get(page, {}).get(z, 0.0) + score

        # Table text snippets used for SEM signal extraction.
        table_text_by_page: Dict[int, List[str]] = {}
        cur.execute(
            """
            SELECT page_number, table_data
            FROM document_tables
            WHERE document_id = ?
            """,
            (doc_id,),
        )
        for page_number, table_data in cur.fetchall():
            td = str(table_data or "")
            if not _SEM_TABLE_KEYWORD_RE.search(td):
                continue
            txt = _table_data_to_text(td)
            if txt:
                table_text_by_page.setdefault(int(page_number), []).append(txt)

        def best_key(scores: Dict[str, float]) -> Optional[str]:
            if not scores:
                return None
            return max(scores.items(), key=lambda kv: kv[1])[0]

        page_to_activity: Dict[int, str] = {}
        activity_geo_map: Dict[str, Dict[str, Optional[str]]] = {}

        # Build activity groups: header page (has "Activity Status:") + following pages until next header
        for gi, start_i in enumerate(header_idxs):
            end_i = header_idxs[gi + 1] if gi + 1 < len(header_idxs) else len(pages)
            group = pages[start_i:end_i]
            header_page, header_text = group[0]

            activity_id = f"{doc_id}:{header_page}"
            for page_num, _txt in group:
                page_to_activity[page_num] = activity_id

            project_number = _extract_project_number(header_text)
            project_title = _extract_project_title(header_text)
            grantee_activity_number = _extract_grantee_activity_number(header_text)

            activity_status, activity_type = _extract_activity_status_and_type(header_text)
            activity_type_group = _activity_type_group(activity_type)

            # Aggregate money stats across pages in this activity group
            money_agg: Dict[str, Dict[str, float]] = {}
            for page_num, _txt in group:
                for lbl, stats in money_by_page.get(page_num, {}).items():
                    a = money_agg.setdefault(lbl, {"n_mentions": 0.0, "sum_amount_usd": 0.0, "max_amount_usd": 0.0})
                    a["n_mentions"] += float(stats.get("n_mentions") or 0.0)
                    a["sum_amount_usd"] += float(stats.get("sum_amount_usd") or 0.0)
                    a["max_amount_usd"] = max(a["max_amount_usd"], float(stats.get("max_amount_usd") or 0.0))

            # Aggregate geography hints across pages in this activity group
            county_scores_direct: Dict[str, float] = {}
            county_scores_any: Dict[str, float] = {}
            city_scores: Dict[str, float] = {}
            zip_scores: Dict[str, float] = {}
            for page_num, _txt in group:
                for fips3, sc in page_county_direct.get(page_num, {}).items():
                    county_scores_direct[fips3] = county_scores_direct.get(fips3, 0.0) + sc
                for fips3, sc in page_county_any.get(page_num, {}).items():
                    county_scores_any[fips3] = county_scores_any.get(fips3, 0.0) + sc
                for c, sc in page_city.get(page_num, {}).items():
                    city_scores[c] = city_scores.get(c, 0.0) + sc
                for z, sc in page_zip.get(page_num, {}).items():
                    zip_scores[z] = zip_scores.get(z, 0.0) + sc

            county_fips3 = best_key(county_scores_direct) or best_key(county_scores_any)
            county_name = county_index.by_fips3.get(county_fips3) if (county_index and county_fips3) else None
            city_name = best_key(city_scores)
            zip5 = best_key(zip_scores)

            # Beneficiary measures (from any page in the group)
            benef_summary: Dict[str, Dict[str, Optional[float]]] = {}
            benef_lmi = None
            group_text = "\n".join([txt for _, txt in group if txt])
            projected_start_date = _extract_first(group_text, r"Projected Start Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})")
            projected_end_date = _extract_first(group_text, r"Projected End Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})")
            completed_end_date = _extract_first(
                group_text,
                r"Completed Activity Actual End Date:\s*([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})",
            )
            duration_quarters_est = _duration_quarters(
                _parse_date_token(projected_start_date),
                _parse_date_token(completed_end_date or projected_end_date),
            )

            for page_num, page_text in group:
                for row in _parse_beneficiary_rows(page_text):
                    beneficiary_rows.append(
                        {
                            "activity_id": activity_id,
                            "document_id": doc_id,
                            "header_page": header_page,
                            "page_number": page_num,
                            **row,
                        }
                    )
                    if row.get("lmi_percent") is not None:
                        benef_lmi = row.get("lmi_percent")

                    canonical = row.get("canonical_measure")
                    if not canonical:
                        continue
                    cur_rec = benef_summary.setdefault(
                        canonical,
                        {"this_period": None, "total_actual": None, "total_expected": None},
                    )
                    # Use max to avoid double-counting across continuation pages
                    for k in ["this_period", "total_actual", "total_expected"]:
                        v = row.get(k)
                        if v is None:
                            continue
                        prev = cur_rec.get(k)
                        if prev is None or (isinstance(v, (int, float)) and v > prev):
                            cur_rec[k] = float(v)

            activity_row: Dict[str, Any] = {
                "activity_id": activity_id,
                "document_id": doc_id,
                "header_page": header_page,
                "start_page": group[0][0],
                "end_page": group[-1][0],
                "n_pages": len(group),
                "filename": doc.filename,
                "category": doc.category,
                "disaster_code": doc.disaster_code,
                "year": doc_year,
                "quarter": doc_quarter,
                "quarter_label": doc.quarter_label,
                "project_number": project_number,
                "project_title": project_title,
                "grantee_activity_number": grantee_activity_number,
                "activity_type": activity_type,
                "activity_type_group": activity_type_group,
                "activity_status": activity_status,
                "geo_county_name": county_name,
                "geo_county_fips3": county_fips3,
                "geo_city": city_name,
                "geo_zip": zip5,
                "projected_start_date": projected_start_date,
                "projected_end_date": projected_end_date,
                "completed_activity_actual_end_date": completed_end_date,
                "program_duration_quarters_est": duration_quarters_est,
                "benef_lmi_percent": benef_lmi,
            }

            # Flatten money aggregates to stable columns
            for lbl in ["budget", "obligated", "drawdown", "expended", "unknown"]:
                stats = money_agg.get(lbl) or {}
                activity_row[f"money_n_mentions_{lbl}"] = int(stats.get("n_mentions") or 0)
                activity_row[f"money_sum_amount_usd_{lbl}"] = float(stats.get("sum_amount_usd") or 0.0)
                activity_row[f"money_max_amount_usd_{lbl}"] = float(stats.get("max_amount_usd") or 0.0)

            # Common beneficiary outputs (wide)
            for mkey in ["persons", "households", "owner_households", "renter_households", "jobs_created", "jobs_retained", "housing_units"]:
                rec = benef_summary.get(mkey) or {}
                activity_row[f"benef_{mkey}_this_period"] = rec.get("this_period")
                activity_row[f"benef_{mkey}_total_actual"] = rec.get("total_actual")
                activity_row[f"benef_{mkey}_total_expected"] = rec.get("total_expected")

            activities_rows.append(activity_row)
            activity_geo_map[activity_id] = {
                "county_fips3": county_fips3,
                "county_name": county_name,
                "city": city_name,
                "zip": zip5,
            }

        # SEM construct signals from text and relevant tables with provenance.
        for page_num, page_text in pages:
            activity_id = page_to_activity.get(page_num)
            county_fips3 = best_key(page_county_direct.get(page_num, {})) or best_key(page_county_any.get(page_num, {}))
            county_name = county_index.by_fips3.get(county_fips3) if (county_index and county_fips3) else None
            city_name = best_key(page_city.get(page_num, {}))
            zip5 = best_key(page_zip.get(page_num, {}))
            if activity_id and activity_id in activity_geo_map:
                geo = activity_geo_map.get(activity_id) or {}
                county_fips3 = geo.get("county_fips3") or county_fips3
                county_name = geo.get("county_name") or county_name
                city_name = geo.get("city") or city_name
                zip5 = geo.get("zip") or zip5

            for sig in _extract_sem_signals_from_text(page_text, source_type="text_page"):
                sem_signal_rows.append(
                    {
                        "document_id": doc_id,
                        "filename": doc.filename,
                        "category": doc.category,
                        "disaster_code": doc.disaster_code,
                        "year": doc_year,
                        "quarter": doc_quarter,
                        "page_number": page_num,
                        "source_type": "text_page",
                        "activity_id": activity_id,
                        "county_fips3": county_fips3,
                        "county_name": county_name,
                        "city": city_name,
                        "zip": zip5,
                        **sig,
                    }
                )

            for table_text in table_text_by_page.get(page_num, []):
                for sig in _extract_sem_signals_from_text(table_text, source_type="table_json"):
                    sem_signal_rows.append(
                        {
                            "document_id": doc_id,
                            "filename": doc.filename,
                            "category": doc.category,
                            "disaster_code": doc.disaster_code,
                            "year": doc_year,
                            "quarter": doc_quarter,
                            "page_number": page_num,
                            "source_type": "table_json",
                            "activity_id": activity_id,
                            "county_fips3": county_fips3,
                            "county_name": county_name,
                            "city": city_name,
                            "zip": zip5,
                            **sig,
                        }
                    )

    activities = pd.DataFrame(activities_rows)
    beneficiary_measures = pd.DataFrame(beneficiary_rows)
    sem_construct_signals = pd.DataFrame(sem_signal_rows)
    activities_unique = pd.DataFrame()

    if not activities.empty:
        def valid_activity_key(val: Any) -> bool:
            if val is None:
                return False
            s = _clean_text(val) or ""
            if not s or len(s) < 3 or len(s) > 120:
                return False
            if s.endswith(":"):
                return False
            low = s.lower()
            if low in {"projected", "overall", "activity", "na", "n/a"}:
                return False
            if re.fullmatch(r"\$?[\d,]+(?:\.\d{2})?", s):
                return False
            if re.fullmatch(r"[\d,]+(?:\.\d+)?", s):
                return False
            return bool(re.search(r"\d", s)) or low.startswith("glo-")

        activities["activity_status"] = activities["activity_status"].fillna("Unknown")
        activities["activity_key"] = activities["grantee_activity_number"].map(
            lambda v: _clean_text(v) if valid_activity_key(v) else None
        )
        activities["activity_key"] = activities["activity_key"].fillna(activities["activity_id"])

        status_rank = {"Not Started": 0, "Under Way": 1, "Cancelled": 2, "Completed": 3, "Unknown": -1}
        inv_rank = {v: k for k, v in status_rank.items()}
        activities["_status_rank"] = activities["activity_status"].map(status_rank).fillna(-1).astype(int)

        numeric_cols = (
            [c for c in activities.columns if c.startswith("money_max_amount_usd_")]
            + [c for c in activities.columns if c.startswith("benef_")]
            + [c for c in ["program_duration_quarters_est"] if c in activities.columns]
        )
        for c in numeric_cols:
            activities[c] = pd.to_numeric(activities[c], errors="coerce")

        def first_non_null(series: pd.Series) -> Any:
            s = series.dropna()
            if s.empty:
                return None
            return s.iloc[0]

        dedup_cols = ["category", "disaster_code", "year", "quarter", "activity_key"]
        agg_spec: Dict[str, Any] = {
            "document_id": "first",
            "filename": "first",
            "activity_id": "first",
            "grantee_activity_number": first_non_null,
            "project_number": first_non_null,
            "project_title": first_non_null,
            "activity_type": first_non_null,
            "activity_type_group": first_non_null,
            "geo_county_fips3": first_non_null,
            "geo_county_name": first_non_null,
            "geo_city": first_non_null,
            "geo_zip": first_non_null,
            "projected_start_date": first_non_null,
            "projected_end_date": first_non_null,
            "completed_activity_actual_end_date": first_non_null,
            "_status_rank": "max",
            "benef_lmi_percent": "max",
        }
        for c in numeric_cols:
            agg_spec[c] = "max"

        activities_unique = (
            activities.sort_values(
                ["category", "disaster_code", "year", "quarter", "activity_key", "_status_rank"],
                ascending=[True, True, True, True, True, False],
            )
            .groupby(dedup_cols, dropna=False)
            .agg(agg_spec)
            .reset_index()
        )
        n_docs = (
            activities.groupby(dedup_cols, dropna=False)["document_id"]
            .nunique()
            .reset_index(name="n_documents")
        )
        activities_unique = activities_unique.merge(n_docs, on=dedup_cols, how="left")
        activities_unique["activity_status"] = activities_unique["_status_rank"].map(inv_rank).fillna("Unknown")
        activities_unique = activities_unique.drop(columns=["_status_rank"])
        activities = activities.drop(columns=["_status_rank"])

    # ---- build panels ----
    # panel_document: documents + money + entity counts + activity counts
    panel_document = documents.merge(money_doc_wide, on="document_id", how="left").merge(
        entity_doc_wide, on="document_id", how="left"
    )
    if not activities.empty:
        act_counts = (
            activities.groupby("document_id")
            .agg(
                n_activities=("activity_id", "count"),
                n_activity_pages=("n_pages", "sum"),
                n_unique_activities=("activity_key", pd.Series.nunique),
            )
            .reset_index()
        )
        panel_document = panel_document.merge(act_counts, on="document_id", how="left")
    panel_document["n_activities"] = panel_document.get("n_activities", 0).fillna(0).astype(int)
    panel_document["n_activity_pages"] = panel_document.get("n_activity_pages", 0).fillna(0).astype(int)
    panel_document["n_unique_activities"] = panel_document.get("n_unique_activities", 0).fillna(0).astype(int)

    # panel_disaster_quarter: aggregate panel_document to quarter
    money_sum_cols = [c for c in panel_document.columns if c.startswith("money_sum_amount_usd_")]
    entity_cols = [c for c in panel_document.columns if c.startswith("entity_")]
    group_cols = ["category", "disaster_code", "year", "quarter"]
    agg_spec: Dict[str, Any] = {
        "document_id": "count",
        "page_count": "sum",
        "file_size_bytes": "sum",
        "n_activity_pages": "sum",
        "n_activities": "sum",
    }
    for c in money_sum_cols:
        agg_spec[c] = "sum"
    for c in entity_cols:
        agg_spec[c] = "sum"
    panel_disaster_quarter = (
        panel_document.groupby(group_cols, dropna=False)
        .agg(agg_spec)
        .rename(columns={"document_id": "n_documents"})
        .reset_index()
        .sort_values(group_cols)
    )

    # Add activity-derived aggregates to panel_disaster_quarter (if available)
    if not activities_unique.empty:
        act = activities_unique.copy()
        act["activity_status"] = act["activity_status"].fillna("Unknown")
        act["activity_type_group"] = act["activity_type_group"].fillna("Unknown")

        money_max_cols = [c for c in act.columns if c.startswith("money_max_amount_usd_")]
        benef_sum_cols = [c for c in act.columns if c.startswith("benef_") and c != "benef_lmi_percent"]
        for c in money_max_cols + benef_sum_cols + ["benef_lmi_percent", "program_duration_quarters_est"]:
            if c in act.columns:
                act[c] = pd.to_numeric(act[c], errors="coerce")

        base = (
            act.groupby(group_cols, dropna=False)
            .agg(
                act_n_unique_activities=("activity_key", "count"),
                act_n_activity_documents=("document_id", pd.Series.nunique),
                act_n_counties=("geo_county_fips3", pd.Series.nunique),
                act_n_cities=("geo_city", pd.Series.nunique),
                act_mean_benef_lmi_percent=("benef_lmi_percent", "mean"),
                act_mean_program_duration_quarters=("program_duration_quarters_est", "mean"),
                act_n_program_duration_observed=("program_duration_quarters_est", "count"),
            )
            .reset_index()
        )
        for col, out_name in [
            ("money_max_amount_usd_budget", "sum_budget_usd"),
            ("money_max_amount_usd_obligated", "sum_obligated_usd"),
            ("money_max_amount_usd_drawdown", "sum_drawdown_usd"),
            ("money_max_amount_usd_expended", "sum_expended_usd"),
        ]:
            if col in act.columns:
                sums = act.groupby(group_cols, dropna=False)[col].sum(min_count=1).reset_index(name=f"act_{out_name}")
                base = base.merge(sums, on=group_cols, how="left")

        for bcol in benef_sum_cols:
            sums = act.groupby(group_cols, dropna=False)[bcol].sum(min_count=1).reset_index(name=f"act_sum_{bcol}")
            base = base.merge(sums, on=group_cols, how="left")

        admin_act = act[act["activity_type_group"] == "Administration"].copy()
        if not admin_act.empty:
            admin_base = (
                admin_act.groupby(group_cols, dropna=False)
                .agg(
                    act_n_admin_activities=("activity_key", "count"),
                    act_admin_sum_budget_usd=("money_max_amount_usd_budget", "sum"),
                    act_admin_sum_obligated_usd=("money_max_amount_usd_obligated", "sum"),
                    act_admin_sum_drawdown_usd=("money_max_amount_usd_drawdown", "sum"),
                    act_admin_sum_expended_usd=("money_max_amount_usd_expended", "sum"),
                )
                .reset_index()
            )
            base = base.merge(admin_base, on=group_cols, how="left")

        status_counts = (
            act.pivot_table(
                index=group_cols,
                columns="activity_status",
                values="activity_key",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        status_cols = [c for c in status_counts.columns if c not in set(group_cols)]
        status_counts = status_counts.rename(columns={c: f"act_n_status_{_normalize_key(str(c))}" for c in status_cols})

        type_counts = (
            act.pivot_table(
                index=group_cols,
                columns="activity_type_group",
                values="activity_key",
                aggfunc="count",
                fill_value=0,
            )
            .reset_index()
        )
        type_cols = [c for c in type_counts.columns if c not in set(group_cols)]
        type_counts = type_counts.rename(columns={c: f"act_n_type_{_normalize_key(str(c))}" for c in type_cols})

        act_quarter = base.merge(status_counts, on=group_cols, how="left").merge(type_counts, on=group_cols, how="left")
        completed_col = "act_n_status_completed"
        if completed_col in act_quarter.columns:
            act_quarter["act_completion_rate"] = act_quarter[completed_col] / act_quarter["act_n_unique_activities"].replace(0, pd.NA)

        panel_disaster_quarter = panel_disaster_quarter.merge(act_quarter, on=group_cols, how="left")

    # Add severity proxy columns (optional) + keep long format exports below
    if not severity_proxies_by_quarter.empty:
        sev = severity_proxies_by_quarter.copy()
        sev_wide = (
            sev.pivot_table(
                index=group_cols,
                columns="metric",
                values=["n_mentions", "max_value"],
                aggfunc="first",
            )
            .reset_index()
        )
        new_cols: List[str] = []
        for col in sev_wide.columns:
            if isinstance(col, tuple) and len(col) == 2:
                top, metric = col
                if metric in ("", None):
                    new_cols.append(str(top))
                else:
                    new_cols.append(f"severity_{metric}_{top}")
            else:
                new_cols.append(str(col))
        sev_wide.columns = new_cols
        panel_disaster_quarter = panel_disaster_quarter.merge(sev_wide, on=group_cols, how="left")

    # County-by-quarter panel (from unique activities with county hints)
    panel_county_quarter = pd.DataFrame()
    panel_city_quarter = pd.DataFrame()
    panel_state_quarter = pd.DataFrame()

    if not activities_unique.empty:
        act = activities_unique.copy()
        act["activity_status"] = act["activity_status"].fillna("Unknown")
        act["activity_type_group"] = act["activity_type_group"].fillna("Unknown")

        geo_group_cols = ["category", "disaster_code", "year", "quarter"]
        county_cols = ["geo_county_name", "geo_county_fips3"]
        city_cols = ["geo_city", "geo_county_fips3", "geo_county_name"]

        # County
        act_county = act.dropna(subset=["geo_county_fips3"]).copy()
        if not act_county.empty:
            sums = (
                act_county.groupby(geo_group_cols + county_cols, dropna=False)
                .agg(
                    n_activities=("activity_key", "count"),
                    n_projects=("project_number", pd.Series.nunique),
                    n_documents=("document_id", pd.Series.nunique),
                    n_cities=("geo_city", pd.Series.nunique),
                    mean_program_duration_quarters=("program_duration_quarters_est", "mean"),
                    n_program_duration_observed=("program_duration_quarters_est", "count"),
                )
                .reset_index()
            )
            for col, out_name in [
                ("money_max_amount_usd_budget", "sum_budget_usd"),
                ("money_max_amount_usd_obligated", "sum_obligated_usd"),
                ("money_max_amount_usd_drawdown", "sum_drawdown_usd"),
                ("money_max_amount_usd_expended", "sum_expended_usd"),
            ]:
                if col in act_county.columns:
                    s = act_county.groupby(geo_group_cols + county_cols, dropna=False)[col].sum(min_count=1).reset_index(name=out_name)
                    sums = sums.merge(s, on=geo_group_cols + county_cols, how="left")

            benef_cols = [c for c in act_county.columns if c.startswith("benef_") and c != "benef_lmi_percent"]
            for bcol in benef_cols:
                s = act_county.groupby(geo_group_cols + county_cols, dropna=False)[bcol].sum(min_count=1).reset_index(name=f"sum_{bcol}")
                sums = sums.merge(s, on=geo_group_cols + county_cols, how="left")

            admin_county = act_county[act_county["activity_type_group"] == "Administration"].copy()
            if not admin_county.empty:
                admin = (
                    admin_county.groupby(geo_group_cols + county_cols, dropna=False)
                    .agg(
                        admin_n_activities=("activity_key", "count"),
                        admin_sum_budget_usd=("money_max_amount_usd_budget", "sum"),
                        admin_sum_obligated_usd=("money_max_amount_usd_obligated", "sum"),
                        admin_sum_drawdown_usd=("money_max_amount_usd_drawdown", "sum"),
                        admin_sum_expended_usd=("money_max_amount_usd_expended", "sum"),
                    )
                    .reset_index()
                )
                sums = sums.merge(admin, on=geo_group_cols + county_cols, how="left")

            status_counts = (
                act_county.pivot_table(
                    index=geo_group_cols + county_cols,
                    columns="activity_status",
                    values="activity_key",
                    aggfunc="count",
                    fill_value=0,
                )
                .reset_index()
            )
            status_cols = [c for c in status_counts.columns if c not in set(geo_group_cols + county_cols)]
            status_counts = status_counts.rename(columns={c: f"n_status_{_normalize_key(str(c))}" for c in status_cols})

            type_counts = (
                act_county.pivot_table(
                    index=geo_group_cols + county_cols,
                    columns="activity_type_group",
                    values="activity_key",
                    aggfunc="count",
                    fill_value=0,
                )
                .reset_index()
            )
            type_cols = [c for c in type_counts.columns if c not in set(geo_group_cols + county_cols)]
            type_counts = type_counts.rename(columns={c: f"n_type_{_normalize_key(str(c))}" for c in type_cols})

            panel_county_quarter = sums.merge(status_counts, on=geo_group_cols + county_cols, how="left").merge(
                type_counts, on=geo_group_cols + county_cols, how="left"
            )
            completed_col = "n_status_completed"
            if completed_col in panel_county_quarter.columns:
                panel_county_quarter["completion_rate"] = panel_county_quarter[completed_col] / panel_county_quarter["n_activities"].replace(0, pd.NA)

            panel_county_quarter = panel_county_quarter.rename(
                columns={"geo_county_name": "county_name", "geo_county_fips3": "county_fips3"}
            ).sort_values(geo_group_cols + ["county_name"])

        # City
        act_city = act.dropna(subset=["geo_city"]).copy()
        if not act_city.empty:
            sums = (
                act_city.groupby(geo_group_cols + city_cols, dropna=False)
                .agg(
                    n_activities=("activity_key", "count"),
                    n_projects=("project_number", pd.Series.nunique),
                    n_documents=("document_id", pd.Series.nunique),
                    mean_program_duration_quarters=("program_duration_quarters_est", "mean"),
                    n_program_duration_observed=("program_duration_quarters_est", "count"),
                )
                .reset_index()
            )
            for col, out_name in [
                ("money_max_amount_usd_budget", "sum_budget_usd"),
                ("money_max_amount_usd_obligated", "sum_obligated_usd"),
                ("money_max_amount_usd_drawdown", "sum_drawdown_usd"),
                ("money_max_amount_usd_expended", "sum_expended_usd"),
            ]:
                if col in act_city.columns:
                    s = act_city.groupby(geo_group_cols + city_cols, dropna=False)[col].sum(min_count=1).reset_index(name=out_name)
                    sums = sums.merge(s, on=geo_group_cols + city_cols, how="left")

            benef_cols = [c for c in act_city.columns if c.startswith("benef_") and c != "benef_lmi_percent"]
            for bcol in benef_cols:
                s = act_city.groupby(geo_group_cols + city_cols, dropna=False)[bcol].sum(min_count=1).reset_index(name=f"sum_{bcol}")
                sums = sums.merge(s, on=geo_group_cols + city_cols, how="left")

            admin_city = act_city[act_city["activity_type_group"] == "Administration"].copy()
            if not admin_city.empty:
                admin = (
                    admin_city.groupby(geo_group_cols + city_cols, dropna=False)
                    .agg(
                        admin_n_activities=("activity_key", "count"),
                        admin_sum_budget_usd=("money_max_amount_usd_budget", "sum"),
                        admin_sum_obligated_usd=("money_max_amount_usd_obligated", "sum"),
                        admin_sum_drawdown_usd=("money_max_amount_usd_drawdown", "sum"),
                        admin_sum_expended_usd=("money_max_amount_usd_expended", "sum"),
                    )
                    .reset_index()
                )
                sums = sums.merge(admin, on=geo_group_cols + city_cols, how="left")

            status_counts = (
                act_city.pivot_table(
                    index=geo_group_cols + city_cols,
                    columns="activity_status",
                    values="activity_key",
                    aggfunc="count",
                    fill_value=0,
                )
                .reset_index()
            )
            status_cols = [c for c in status_counts.columns if c not in set(geo_group_cols + city_cols)]
            status_counts = status_counts.rename(columns={c: f"n_status_{_normalize_key(str(c))}" for c in status_cols})

            type_counts = (
                act_city.pivot_table(
                    index=geo_group_cols + city_cols,
                    columns="activity_type_group",
                    values="activity_key",
                    aggfunc="count",
                    fill_value=0,
                )
                .reset_index()
            )
            type_cols = [c for c in type_counts.columns if c not in set(geo_group_cols + city_cols)]
            type_counts = type_counts.rename(columns={c: f"n_type_{_normalize_key(str(c))}" for c in type_cols})

            panel_city_quarter = (
                sums.merge(status_counts, on=geo_group_cols + city_cols, how="left")
                .merge(type_counts, on=geo_group_cols + city_cols, how="left")
                .rename(columns={"geo_city": "city", "geo_county_name": "county_name", "geo_county_fips3": "county_fips3"})
            )
            completed_col = "n_status_completed"
            if completed_col in panel_city_quarter.columns:
                panel_city_quarter["completion_rate"] = panel_city_quarter[completed_col] / panel_city_quarter["n_activities"].replace(0, pd.NA)
            panel_city_quarter = panel_city_quarter.sort_values(geo_group_cols + ["city"])

        # State (year/quarter rollup across all disasters)
        if "year" in act.columns and "quarter" in act.columns:
            act_state = act.copy()
            act_state["disaster_key"] = act_state["category"].astype(str) + ":" + act_state["disaster_code"].astype(str)
            panel_state_quarter = (
                act_state.groupby(["year", "quarter"], dropna=False)
                .agg(
                    n_unique_activities=("activity_key", "count"),
                    n_disasters=("disaster_key", pd.Series.nunique),
                    n_documents=("document_id", pd.Series.nunique),
                    n_counties=("geo_county_fips3", pd.Series.nunique),
                    n_cities=("geo_city", pd.Series.nunique),
                    sum_budget_usd=("money_max_amount_usd_budget", "sum"),
                    sum_obligated_usd=("money_max_amount_usd_obligated", "sum"),
                    sum_drawdown_usd=("money_max_amount_usd_drawdown", "sum"),
                    sum_expended_usd=("money_max_amount_usd_expended", "sum"),
                    mean_program_duration_quarters=("program_duration_quarters_est", "mean"),
                    n_program_duration_observed=("program_duration_quarters_est", "count"),
                )
                .reset_index()
                .sort_values(["year", "quarter"])
            )
            admin_state = act_state[act_state["activity_type_group"] == "Administration"].copy()
            if not admin_state.empty:
                admin = (
                    admin_state.groupby(["year", "quarter"], dropna=False)
                    .agg(
                        admin_n_activities=("activity_key", "count"),
                        admin_sum_budget_usd=("money_max_amount_usd_budget", "sum"),
                        admin_sum_obligated_usd=("money_max_amount_usd_obligated", "sum"),
                        admin_sum_drawdown_usd=("money_max_amount_usd_drawdown", "sum"),
                        admin_sum_expended_usd=("money_max_amount_usd_expended", "sum"),
                    )
                    .reset_index()
                )
                panel_state_quarter = panel_state_quarter.merge(admin, on=["year", "quarter"], how="left")
            # Beneficiary sums (where present)
            benef_cols = [c for c in act_state.columns if c.startswith("benef_") and c != "benef_lmi_percent"]
            for bcol in benef_cols:
                s = act_state.groupby(["year", "quarter"], dropna=False)[bcol].sum(min_count=1).reset_index(name=f"sum_{bcol}")
                panel_state_quarter = panel_state_quarter.merge(s, on=["year", "quarter"], how="left")

            # Status and type counts
            status_counts = (
                act_state.pivot_table(
                    index=["year", "quarter"],
                    columns="activity_status",
                    values="activity_key",
                    aggfunc="count",
                    fill_value=0,
                )
                .reset_index()
            )
            status_cols = [c for c in status_counts.columns if c not in {"year", "quarter"}]
            status_counts = status_counts.rename(columns={c: f"n_status_{_normalize_key(str(c))}" for c in status_cols})

            type_counts = (
                act_state.pivot_table(
                    index=["year", "quarter"],
                    columns="activity_type_group",
                    values="activity_key",
                    aggfunc="count",
                    fill_value=0,
                )
                .reset_index()
            )
            type_cols = [c for c in type_counts.columns if c not in {"year", "quarter"}]
            type_counts = type_counts.rename(columns={c: f"n_type_{_normalize_key(str(c))}" for c in type_cols})

            panel_state_quarter = panel_state_quarter.merge(status_counts, on=["year", "quarter"], how="left").merge(
                type_counts, on=["year", "quarter"], how="left"
            )
            completed_col = "n_status_completed"
            if completed_col in panel_state_quarter.columns:
                panel_state_quarter["completion_rate"] = panel_state_quarter[completed_col] / panel_state_quarter["n_unique_activities"].replace(0, pd.NA)

    # ---- SEM-oriented panels/signals ----
    sem_signal_metrics = [
        "admin_staff_count",
        "admin_payroll_usd",
        "affected_population_persons",
        "affected_population_households",
        "severity_deaths_count",
        "severity_economic_loss_usd",
        "severity_property_damage_usd",
        "severity_unmet_need_usd",
    ]

    if not sem_construct_signals.empty:
        sem_construct_signals["value"] = pd.to_numeric(sem_construct_signals["value"], errors="coerce")
        sem_construct_signals["confidence"] = pd.to_numeric(sem_construct_signals["confidence"], errors="coerce")
        sem_construct_signals = sem_construct_signals.dropna(subset=["value", "metric"])
        sem_construct_signals = sem_construct_signals[sem_construct_signals["metric"].isin(sem_signal_metrics)].copy()

    def sem_signal_wide(signals: pd.DataFrame, group_cols_in: List[str]) -> pd.DataFrame:
        if signals.empty:
            cols = group_cols_in + ["sem_signal_count", "sem_signal_confidence_mean"]
            for metric in sem_signal_metrics:
                cols.extend([f"{metric}_sum", f"{metric}_n_obs", f"{metric}_confidence_mean"])
            return pd.DataFrame(columns=cols)

        grp = (
            signals.groupby(group_cols_in + ["metric"], dropna=False)
            .agg(value_sum=("value", "sum"), n_obs=("value", "count"), confidence_mean=("confidence", "mean"))
            .reset_index()
        )
        value_wide = grp.pivot_table(index=group_cols_in, columns="metric", values="value_sum", aggfunc="first").reset_index()
        n_wide = grp.pivot_table(index=group_cols_in, columns="metric", values="n_obs", aggfunc="first").reset_index()
        c_wide = grp.pivot_table(index=group_cols_in, columns="metric", values="confidence_mean", aggfunc="first").reset_index()

        value_wide = value_wide.rename(columns={m: f"{m}_sum" for m in value_wide.columns if m not in group_cols_in})
        n_wide = n_wide.rename(columns={m: f"{m}_n_obs" for m in n_wide.columns if m not in group_cols_in})
        c_wide = c_wide.rename(columns={m: f"{m}_confidence_mean" for m in c_wide.columns if m not in group_cols_in})

        overall = (
            signals.groupby(group_cols_in, dropna=False)
            .agg(sem_signal_count=("value", "count"), sem_signal_confidence_mean=("confidence", "mean"))
            .reset_index()
        )
        out = overall.merge(value_wide, on=group_cols_in, how="left").merge(n_wide, on=group_cols_in, how="left").merge(c_wide, on=group_cols_in, how="left")
        for metric in sem_signal_metrics:
            for suffix in ["sum", "n_obs", "confidence_mean"]:
                col = f"{metric}_{suffix}"
                if col not in out.columns:
                    out[col] = pd.NA
        ordered = list(group_cols_in) + ["sem_signal_count", "sem_signal_confidence_mean"]
        for metric in sem_signal_metrics:
            ordered.extend([f"{metric}_sum", f"{metric}_n_obs", f"{metric}_confidence_mean"])
        keep = [c for c in ordered if c in out.columns] + [c for c in out.columns if c not in ordered]
        out = out[keep]
        return out

    sem_disaster_signals = sem_signal_wide(sem_construct_signals, ["category", "disaster_code", "year", "quarter"])
    sem_county_signals = sem_signal_wide(
        sem_construct_signals.dropna(subset=["county_fips3"]),
        ["category", "disaster_code", "year", "quarter", "county_fips3"],
    )
    sem_city_signals = sem_signal_wide(
        sem_construct_signals.dropna(subset=["city"]),
        ["category", "disaster_code", "year", "quarter", "city", "county_fips3"],
    )
    sem_state_signals = sem_signal_wide(sem_construct_signals, ["year", "quarter"])

    panel_disaster_quarter_sem = pd.DataFrame()
    if not panel_disaster_quarter.empty:
        panel_disaster_quarter_sem = panel_disaster_quarter[["category", "disaster_code", "year", "quarter"]].copy()
        dmap = {
            "n_documents": "n_documents",
            "act_n_unique_activities": "programs_total",
            "act_n_status_completed": "programs_completed",
            "act_n_status_under_way": "programs_in_progress",
            "act_n_status_cancelled": "programs_cancelled",
            "act_completion_rate": "program_completion_rate",
            "act_mean_program_duration_quarters": "program_duration_quarters_mean",
            "act_n_program_duration_observed": "program_duration_quarters_n_obs",
            "act_sum_budget_usd": "sum_budget_usd",
            "act_sum_obligated_usd": "sum_obligated_usd",
            "act_sum_drawdown_usd": "sum_drawdown_usd",
            "act_sum_expended_usd": "sum_expended_usd",
            "act_n_admin_activities": "admin_activity_count",
            "act_admin_sum_budget_usd": "admin_activity_budget_usd",
            "act_admin_sum_obligated_usd": "admin_activity_obligated_usd",
            "act_admin_sum_drawdown_usd": "admin_activity_drawdown_usd",
            "act_admin_sum_expended_usd": "admin_activity_expended_usd",
            "act_sum_benef_persons_total_actual": "outcome_persons_total_actual",
            "act_sum_benef_persons_total_expected": "outcome_persons_total_expected",
            "act_sum_benef_households_total_actual": "outcome_households_total_actual",
            "act_sum_benef_households_total_expected": "outcome_households_total_expected",
            "act_sum_benef_owner_households_total_actual": "outcome_owner_households_total_actual",
            "act_sum_benef_owner_households_total_expected": "outcome_owner_households_total_expected",
            "act_sum_benef_renter_households_total_actual": "outcome_renter_households_total_actual",
            "act_sum_benef_renter_households_total_expected": "outcome_renter_households_total_expected",
            "act_sum_benef_jobs_created_total_actual": "outcome_jobs_created_total_actual",
            "act_sum_benef_jobs_created_total_expected": "outcome_jobs_created_total_expected",
            "act_sum_benef_jobs_retained_total_actual": "outcome_jobs_retained_total_actual",
            "act_sum_benef_jobs_retained_total_expected": "outcome_jobs_retained_total_expected",
            "act_sum_benef_housing_units_total_actual": "outcome_housing_units_total_actual",
            "act_sum_benef_housing_units_total_expected": "outcome_housing_units_total_expected",
            "severity_rainfall_inches_max_value": "severity_rainfall_inches_max",
            "severity_wind_speed_mph_max_value": "severity_wind_speed_mph_max",
        }
        for src_col, out_col in dmap.items():
            if src_col in panel_disaster_quarter.columns:
                panel_disaster_quarter_sem[out_col] = panel_disaster_quarter[src_col]
        panel_disaster_quarter_sem = panel_disaster_quarter_sem.merge(
            sem_disaster_signals,
            on=["category", "disaster_code", "year", "quarter"],
            how="left",
        )

    panel_county_quarter_sem = pd.DataFrame()
    if not panel_county_quarter.empty:
        panel_county_quarter_sem = panel_county_quarter[
            ["category", "disaster_code", "year", "quarter", "county_name", "county_fips3"]
        ].copy()
        cmap = {
            "n_activities": "programs_total",
            "n_status_completed": "programs_completed",
            "n_status_under_way": "programs_in_progress",
            "n_status_cancelled": "programs_cancelled",
            "completion_rate": "program_completion_rate",
            "mean_program_duration_quarters": "program_duration_quarters_mean",
            "n_program_duration_observed": "program_duration_quarters_n_obs",
            "sum_budget_usd": "sum_budget_usd",
            "sum_obligated_usd": "sum_obligated_usd",
            "sum_drawdown_usd": "sum_drawdown_usd",
            "sum_expended_usd": "sum_expended_usd",
            "admin_n_activities": "admin_activity_count",
            "admin_sum_budget_usd": "admin_activity_budget_usd",
            "admin_sum_obligated_usd": "admin_activity_obligated_usd",
            "admin_sum_drawdown_usd": "admin_activity_drawdown_usd",
            "admin_sum_expended_usd": "admin_activity_expended_usd",
            "sum_benef_persons_total_actual": "outcome_persons_total_actual",
            "sum_benef_persons_total_expected": "outcome_persons_total_expected",
            "sum_benef_households_total_actual": "outcome_households_total_actual",
            "sum_benef_households_total_expected": "outcome_households_total_expected",
            "sum_benef_owner_households_total_actual": "outcome_owner_households_total_actual",
            "sum_benef_owner_households_total_expected": "outcome_owner_households_total_expected",
            "sum_benef_renter_households_total_actual": "outcome_renter_households_total_actual",
            "sum_benef_renter_households_total_expected": "outcome_renter_households_total_expected",
            "sum_benef_jobs_created_total_actual": "outcome_jobs_created_total_actual",
            "sum_benef_jobs_created_total_expected": "outcome_jobs_created_total_expected",
            "sum_benef_jobs_retained_total_actual": "outcome_jobs_retained_total_actual",
            "sum_benef_jobs_retained_total_expected": "outcome_jobs_retained_total_expected",
            "sum_benef_housing_units_total_actual": "outcome_housing_units_total_actual",
            "sum_benef_housing_units_total_expected": "outcome_housing_units_total_expected",
        }
        for src_col, out_col in cmap.items():
            if src_col in panel_county_quarter.columns:
                panel_county_quarter_sem[out_col] = panel_county_quarter[src_col]
        panel_county_quarter_sem = panel_county_quarter_sem.merge(
            sem_county_signals,
            on=["category", "disaster_code", "year", "quarter", "county_fips3"],
            how="left",
        )

    panel_city_quarter_sem = pd.DataFrame()
    if not panel_city_quarter.empty:
        panel_city_quarter_sem = panel_city_quarter[
            ["category", "disaster_code", "year", "quarter", "city", "county_fips3", "county_name"]
        ].copy()
        cmap = {
            "n_activities": "programs_total",
            "n_status_completed": "programs_completed",
            "n_status_under_way": "programs_in_progress",
            "n_status_cancelled": "programs_cancelled",
            "completion_rate": "program_completion_rate",
            "mean_program_duration_quarters": "program_duration_quarters_mean",
            "n_program_duration_observed": "program_duration_quarters_n_obs",
            "sum_budget_usd": "sum_budget_usd",
            "sum_obligated_usd": "sum_obligated_usd",
            "sum_drawdown_usd": "sum_drawdown_usd",
            "sum_expended_usd": "sum_expended_usd",
            "admin_n_activities": "admin_activity_count",
            "admin_sum_budget_usd": "admin_activity_budget_usd",
            "admin_sum_obligated_usd": "admin_activity_obligated_usd",
            "admin_sum_drawdown_usd": "admin_activity_drawdown_usd",
            "admin_sum_expended_usd": "admin_activity_expended_usd",
            "sum_benef_persons_total_actual": "outcome_persons_total_actual",
            "sum_benef_persons_total_expected": "outcome_persons_total_expected",
            "sum_benef_households_total_actual": "outcome_households_total_actual",
            "sum_benef_households_total_expected": "outcome_households_total_expected",
            "sum_benef_owner_households_total_actual": "outcome_owner_households_total_actual",
            "sum_benef_owner_households_total_expected": "outcome_owner_households_total_expected",
            "sum_benef_renter_households_total_actual": "outcome_renter_households_total_actual",
            "sum_benef_renter_households_total_expected": "outcome_renter_households_total_expected",
            "sum_benef_jobs_created_total_actual": "outcome_jobs_created_total_actual",
            "sum_benef_jobs_created_total_expected": "outcome_jobs_created_total_expected",
            "sum_benef_jobs_retained_total_actual": "outcome_jobs_retained_total_actual",
            "sum_benef_jobs_retained_total_expected": "outcome_jobs_retained_total_expected",
            "sum_benef_housing_units_total_actual": "outcome_housing_units_total_actual",
            "sum_benef_housing_units_total_expected": "outcome_housing_units_total_expected",
        }
        for src_col, out_col in cmap.items():
            if src_col in panel_city_quarter.columns:
                panel_city_quarter_sem[out_col] = panel_city_quarter[src_col]
        panel_city_quarter_sem = panel_city_quarter_sem.merge(
            sem_city_signals,
            on=["category", "disaster_code", "year", "quarter", "city", "county_fips3"],
            how="left",
        )

    panel_state_quarter_sem = pd.DataFrame()
    if not panel_state_quarter.empty:
        panel_state_quarter_sem = panel_state_quarter[["year", "quarter"]].copy()
        smap = {
            "n_unique_activities": "programs_total",
            "n_status_completed": "programs_completed",
            "n_status_under_way": "programs_in_progress",
            "n_status_cancelled": "programs_cancelled",
            "completion_rate": "program_completion_rate",
            "mean_program_duration_quarters": "program_duration_quarters_mean",
            "n_program_duration_observed": "program_duration_quarters_n_obs",
            "sum_budget_usd": "sum_budget_usd",
            "sum_obligated_usd": "sum_obligated_usd",
            "sum_drawdown_usd": "sum_drawdown_usd",
            "sum_expended_usd": "sum_expended_usd",
            "admin_n_activities": "admin_activity_count",
            "admin_sum_budget_usd": "admin_activity_budget_usd",
            "admin_sum_obligated_usd": "admin_activity_obligated_usd",
            "admin_sum_drawdown_usd": "admin_activity_drawdown_usd",
            "admin_sum_expended_usd": "admin_activity_expended_usd",
            "sum_benef_persons_total_actual": "outcome_persons_total_actual",
            "sum_benef_persons_total_expected": "outcome_persons_total_expected",
            "sum_benef_households_total_actual": "outcome_households_total_actual",
            "sum_benef_households_total_expected": "outcome_households_total_expected",
            "sum_benef_owner_households_total_actual": "outcome_owner_households_total_actual",
            "sum_benef_owner_households_total_expected": "outcome_owner_households_total_expected",
            "sum_benef_renter_households_total_actual": "outcome_renter_households_total_actual",
            "sum_benef_renter_households_total_expected": "outcome_renter_households_total_expected",
            "sum_benef_jobs_created_total_actual": "outcome_jobs_created_total_actual",
            "sum_benef_jobs_created_total_expected": "outcome_jobs_created_total_expected",
            "sum_benef_jobs_retained_total_actual": "outcome_jobs_retained_total_actual",
            "sum_benef_jobs_retained_total_expected": "outcome_jobs_retained_total_expected",
            "sum_benef_housing_units_total_actual": "outcome_housing_units_total_actual",
            "sum_benef_housing_units_total_expected": "outcome_housing_units_total_expected",
        }
        for src_col, out_col in smap.items():
            if src_col in panel_state_quarter.columns:
                panel_state_quarter_sem[out_col] = panel_state_quarter[src_col]
        panel_state_quarter_sem = panel_state_quarter_sem.merge(
            sem_state_signals,
            on=["year", "quarter"],
            how="left",
        )

    # SEM coverage report: non-null/non-zero rates by variable and panel.
    sem_cov_rows: List[Dict[str, Any]] = []
    sem_panels_for_cov = {
        "panel_disaster_quarter_sem": panel_disaster_quarter_sem,
        "panel_county_quarter_sem": panel_county_quarter_sem,
        "panel_city_quarter_sem": panel_city_quarter_sem,
        "panel_state_quarter_sem": panel_state_quarter_sem,
    }
    key_cols = {"category", "disaster_code", "year", "quarter", "county_name", "county_fips3", "city", "n_documents"}
    for pname, pdf in sem_panels_for_cov.items():
        if pdf.empty:
            continue
        for col in pdf.columns:
            if col in key_cols:
                continue
            s = pd.to_numeric(pdf[col], errors="coerce")
            non_null = int(s.notna().sum())
            non_zero = int((s.fillna(0) != 0).sum())
            sem_cov_rows.append(
                {
                    "panel": pname,
                    "column": col,
                    "rows": int(len(pdf)),
                    "non_null_rows": non_null,
                    "non_zero_rows": non_zero,
                    "pct_non_null": round((non_null / len(pdf)) * 100.0, 2) if len(pdf) else 0.0,
                    "pct_non_zero": round((non_zero / len(pdf)) * 100.0, 2) if len(pdf) else 0.0,
                }
            )
    sem_coverage_report = pd.DataFrame(sem_cov_rows)

    def sem_meta(column: str) -> Tuple[str, str, str]:
        if column in {"category", "disaster_code", "county_name", "county_fips3", "city"}:
            return ("id", "Panel key used for joins and grouping.", "Directly from document/activity metadata.")
        if column in {"year", "quarter"}:
            return ("time", "Calendar quarter key.", "Parsed from report filenames in the documents table.")
        if column == "n_documents":
            return ("count", "Number of source reports in the panel cell.", "Count distinct documents in panel key.")
        if column.startswith("programs_"):
            return ("count", "Program status/count metric for the panel cell.", "Derived from normalized activity status counts.")
        if column == "program_completion_rate":
            return ("ratio_0_1", "Completed programs divided by total programs.", "programs_completed / programs_total.")
        if column.startswith("program_duration_quarters"):
            return (
                "quarters" if column.endswith("_mean") else "count",
                "Program duration summary where projected/completed dates exist.",
                "Estimated from projected start to projected/completed end dates in activity text.",
            )
        if column.startswith("sum_") and column.endswith("_usd"):
            return ("usd", "Financial rollup from activity-level money fields.", "Summed activity-level max money mentions by panel key.")
        if column.startswith("admin_activity_") and column.endswith("_usd"):
            return ("usd", "Administration activity financial rollup.", "Subset where activity_type_group = Administration.")
        if column == "admin_activity_count":
            return ("count", "Number of administration activities.", "Count of activities where activity_type_group = Administration.")
        if column.startswith("outcome_"):
            unit = "count"
            if "household" in column:
                unit = "households"
            elif "persons" in column:
                unit = "persons"
            elif "jobs" in column:
                unit = "jobs"
            elif "housing_units" in column:
                unit = "units"
            return (
                unit,
                "Beneficiary/accomplishment outcome rollup.",
                "Parsed DRGR beneficiary/accomplishment rows aggregated from activities.",
            )
        if column.startswith("severity_") and column.endswith("_max"):
            return ("value", "Maximum weather severity proxy in panel cell.", "From entity-derived rainfall/wind proxies.")
        if column in {"sem_signal_count", "sem_signal_confidence_mean"}:
            return (
                "count" if column.endswith("_count") else "0_to_1",
                "Summary of extracted SEM signal observations in panel cell.",
                "Aggregated from sem_construct_signals rows.",
            )
        if column.endswith("_sum"):
            unit = "usd" if "_usd_" in column or column.endswith("_usd_sum") else "count"
            return (
                unit,
                "Aggregated SEM construct signal value.",
                "Summed extracted construct values from sem_construct_signals.",
            )
        if column.endswith("_n_obs"):
            return ("count", "Number of extracted observations for this SEM construct.", "Count of sem_construct_signals rows in panel cell.")
        if column.endswith("_confidence_mean"):
            return ("0_to_1", "Average confidence score for this SEM construct.", "Mean confidence across sem_construct_signals rows.")
        return ("value", "SEM/model-ready field.", "Derived in scripts/build_model_ready_datasets.py.")

    sem_dict_rows: List[Dict[str, str]] = []
    sem_datasets = {
        "panel_disaster_quarter_sem": panel_disaster_quarter_sem,
        "panel_county_quarter_sem": panel_county_quarter_sem,
        "panel_city_quarter_sem": panel_city_quarter_sem,
        "panel_state_quarter_sem": panel_state_quarter_sem,
    }
    for dname, ddf in sem_datasets.items():
        if ddf.empty:
            continue
        for col in ddf.columns:
            unit, description, derivation = sem_meta(col)
            sem_dict_rows.append(
                {
                    "dataset": dname,
                    "column": col,
                    "unit": unit,
                    "description": description,
                    "derivation": derivation,
                }
            )

    signal_defs = {
        "document_id": ("id", "Source document identifier.", "From documents.id."),
        "filename": ("text", "Source filename.", "From documents.filename."),
        "category": ("id", "Source category.", "From documents.category."),
        "disaster_code": ("id", "Parsed disaster code.", "From documents.disaster_code."),
        "year": ("time", "Source year.", "From parsed filename metadata."),
        "quarter": ("time", "Source quarter.", "From parsed filename metadata."),
        "page_number": ("count", "Source page number.", "From document_text/document_tables page."),
        "source_type": ("category", "Signal source layer.", "text_page or table_json."),
        "activity_id": ("id", "Matched activity block identifier if available.", "Derived from activity header grouping."),
        "county_fips3": ("id", "Inferred county FIPS (3-digit).", "Activity/page-level location inference."),
        "county_name": ("text", "Inferred county name.", "Lookup from county FIPS mapping."),
        "city": ("text", "Inferred city name.", "Best-confidence city mention."),
        "zip": ("id", "Inferred ZIP code.", "Best-confidence ZIP mention."),
        "metric": ("category", "SEM construct metric label.", "Regex classification rule."),
        "value": ("count_or_usd", "Extracted numeric value.", "Numeric parser with currency/count normalization."),
        "unit": ("category", "Value unit.", "Count or USD based on metric."),
        "confidence": ("0_to_1", "Heuristic extraction confidence.", "Pattern-level confidence with source penalty."),
        "method": ("text", "Extractor method tag.", "Rule family + source type."),
        "snippet": ("text", "Source snippet for audit.", "Truncated line-level provenance text."),
    }
    for col, (unit, description, derivation) in signal_defs.items():
        sem_dict_rows.append(
            {
                "dataset": "sem_construct_signals",
                "column": col,
                "unit": unit,
                "description": description,
                "derivation": derivation,
            }
        )

    sem_variable_dictionary = pd.DataFrame(sem_dict_rows).drop_duplicates(subset=["dataset", "column"]).sort_values(
        ["dataset", "column"]
    )

    # ---- write outputs ----
    out_files: Dict[str, Path] = {
        "panel_document": out_dir / "panels" / "panel_document.csv",
        "panel_disaster_quarter": out_dir / "panels" / "panel_disaster_quarter.csv",
        "activities": out_dir / "long" / "activities.csv",
        "activities_unique": out_dir / "long" / "activities_unique.csv",
        "panel_county_quarter": out_dir / "panels" / "panel_county_quarter.csv",
        "panel_city_quarter": out_dir / "panels" / "panel_city_quarter.csv",
        "panel_state_quarter": out_dir / "panels" / "panel_state_quarter.csv",
        "panel_disaster_quarter_sem": out_dir / "panels" / "panel_disaster_quarter_sem.csv",
        "panel_county_quarter_sem": out_dir / "panels" / "panel_county_quarter_sem.csv",
        "panel_city_quarter_sem": out_dir / "panels" / "panel_city_quarter_sem.csv",
        "panel_state_quarter_sem": out_dir / "panels" / "panel_state_quarter_sem.csv",
        "money_mentions_by_quarter": out_dir / "long" / "money_mentions_by_quarter.csv",
        "topic_trends_by_quarter": out_dir / "long" / "topic_trends_by_quarter.csv",
        "entity_counts_by_quarter": out_dir / "long" / "entity_counts_by_quarter.csv",
        "keyword_pages_by_quarter": out_dir / "long" / "keyword_pages_by_quarter.csv",
        "beneficiary_measures": out_dir / "long" / "beneficiary_measures.csv",
        "sem_construct_signals": out_dir / "long" / "sem_construct_signals.csv",
        "severity_proxies_by_quarter": out_dir / "long" / "severity_proxies_by_quarter.csv",
        "fema_declarations_by_quarter": out_dir / "long" / "fema_declarations_by_quarter.csv",
        "sem_coverage_report": out_dir / "meta" / "sem_coverage_report.csv",
        "sem_variable_dictionary": out_dir / "meta" / "sem_variable_dictionary.csv",
    }

    panel_document.to_csv(out_files["panel_document"], index=False)
    panel_disaster_quarter.to_csv(out_files["panel_disaster_quarter"], index=False)
    money_by_quarter = _read_sql(
        con,
        """
        SELECT
          d.category,
          d.disaster_code,
          d.year,
          d.quarter,
          m.context_label,
          COUNT(*) AS n_mentions,
          SUM(m.amount_usd) AS sum_amount_usd
        FROM money_mentions m
        JOIN documents d ON d.id = m.document_id
        GROUP BY d.category, d.disaster_code, d.year, d.quarter, m.context_label
        ORDER BY d.category, d.year, d.quarter, m.context_label
        """,
    )
    money_by_quarter.to_csv(out_files["money_mentions_by_quarter"], index=False)
    topic_by_quarter.to_csv(out_files["topic_trends_by_quarter"], index=False)
    entity_by_quarter.to_csv(out_files["entity_counts_by_quarter"], index=False)
    keyword_pages_by_quarter.to_csv(out_files["keyword_pages_by_quarter"], index=False)

    if activities.empty:
        # Write empty CSV with headers for stability
        pd.DataFrame(columns=["activity_id", "document_id", "header_page"]).to_csv(out_files["activities"], index=False)
    else:
        activities.to_csv(out_files["activities"], index=False)

    if activities_unique.empty:
        pd.DataFrame(columns=["category", "disaster_code", "year", "quarter", "activity_key"]).to_csv(
            out_files["activities_unique"], index=False
        )
    else:
        activities_unique.to_csv(out_files["activities_unique"], index=False)

    if panel_county_quarter.empty:
        pd.DataFrame(columns=["category", "disaster_code", "year", "quarter", "county_name", "county_fips3"]).to_csv(
            out_files["panel_county_quarter"], index=False
        )
    else:
        panel_county_quarter.to_csv(out_files["panel_county_quarter"], index=False)

    if panel_city_quarter.empty:
        pd.DataFrame(columns=["category", "disaster_code", "year", "quarter", "city"]).to_csv(
            out_files["panel_city_quarter"], index=False
        )
    else:
        panel_city_quarter.to_csv(out_files["panel_city_quarter"], index=False)

    if panel_state_quarter.empty:
        pd.DataFrame(columns=["year", "quarter"]).to_csv(out_files["panel_state_quarter"], index=False)
    else:
        panel_state_quarter.to_csv(out_files["panel_state_quarter"], index=False)

    if panel_disaster_quarter_sem.empty:
        pd.DataFrame(columns=["category", "disaster_code", "year", "quarter"]).to_csv(
            out_files["panel_disaster_quarter_sem"], index=False
        )
    else:
        panel_disaster_quarter_sem.to_csv(out_files["panel_disaster_quarter_sem"], index=False)

    if panel_county_quarter_sem.empty:
        pd.DataFrame(columns=["category", "disaster_code", "year", "quarter", "county_fips3"]).to_csv(
            out_files["panel_county_quarter_sem"], index=False
        )
    else:
        panel_county_quarter_sem.to_csv(out_files["panel_county_quarter_sem"], index=False)

    if panel_city_quarter_sem.empty:
        pd.DataFrame(columns=["category", "disaster_code", "year", "quarter", "city"]).to_csv(
            out_files["panel_city_quarter_sem"], index=False
        )
    else:
        panel_city_quarter_sem.to_csv(out_files["panel_city_quarter_sem"], index=False)

    if panel_state_quarter_sem.empty:
        pd.DataFrame(columns=["year", "quarter"]).to_csv(out_files["panel_state_quarter_sem"], index=False)
    else:
        panel_state_quarter_sem.to_csv(out_files["panel_state_quarter_sem"], index=False)

    if beneficiary_measures.empty:
        pd.DataFrame(columns=["activity_id", "document_id", "page_number", "canonical_measure"]).to_csv(
            out_files["beneficiary_measures"], index=False
        )
    else:
        beneficiary_measures.to_csv(out_files["beneficiary_measures"], index=False)

    if sem_construct_signals.empty:
        pd.DataFrame(
            columns=[
                "document_id",
                "filename",
                "category",
                "disaster_code",
                "year",
                "quarter",
                "page_number",
                "source_type",
                "activity_id",
                "county_fips3",
                "county_name",
                "city",
                "zip",
                "metric",
                "value",
                "unit",
                "confidence",
                "method",
                "snippet",
            ]
        ).to_csv(out_files["sem_construct_signals"], index=False)
    else:
        sem_construct_signals.to_csv(out_files["sem_construct_signals"], index=False)

    if severity_proxies_by_quarter.empty:
        pd.DataFrame(columns=["category", "disaster_code", "year", "quarter", "metric", "n_mentions", "max_value"]).to_csv(
            out_files["severity_proxies_by_quarter"], index=False
        )
    else:
        severity_proxies_by_quarter.to_csv(out_files["severity_proxies_by_quarter"], index=False)

    if fema_declarations_by_quarter.empty:
        pd.DataFrame(columns=["category", "disaster_code", "year", "quarter", "fema_number", "n_mentions", "n_documents"]).to_csv(
            out_files["fema_declarations_by_quarter"], index=False
        )
    else:
        fema_declarations_by_quarter.to_csv(out_files["fema_declarations_by_quarter"], index=False)

    if sem_coverage_report.empty:
        pd.DataFrame(
            columns=["panel", "column", "rows", "non_null_rows", "non_zero_rows", "pct_non_null", "pct_non_zero"]
        ).to_csv(out_files["sem_coverage_report"], index=False)
    else:
        sem_coverage_report.to_csv(out_files["sem_coverage_report"], index=False)

    sem_variable_dictionary.to_csv(out_files["sem_variable_dictionary"], index=False)

    manifest = {
        "built_at": _now_iso(),
        "db_path": str(db_path),
        "outputs": {k: str(v.relative_to(ROOT)) for k, v in out_files.items()},
        "row_counts": {
            "panel_document": int(len(panel_document)),
            "panel_disaster_quarter": int(len(panel_disaster_quarter)),
            "activities": int(len(activities)),
            "activities_unique": int(len(activities_unique)),
            "panel_county_quarter": int(len(panel_county_quarter)),
            "panel_city_quarter": int(len(panel_city_quarter)),
            "panel_state_quarter": int(len(panel_state_quarter)),
            "panel_disaster_quarter_sem": int(len(panel_disaster_quarter_sem)),
            "panel_county_quarter_sem": int(len(panel_county_quarter_sem)),
            "panel_city_quarter_sem": int(len(panel_city_quarter_sem)),
            "panel_state_quarter_sem": int(len(panel_state_quarter_sem)),
            "money_mentions_by_quarter": int(len(money_by_quarter)),
            "topic_trends_by_quarter": int(len(topic_by_quarter)),
            "entity_counts_by_quarter": int(len(entity_by_quarter)),
            "keyword_pages_by_quarter": int(len(keyword_pages_by_quarter)),
            "beneficiary_measures": int(len(beneficiary_measures)),
            "sem_construct_signals": int(len(sem_construct_signals)),
            "severity_proxies_by_quarter": int(len(severity_proxies_by_quarter)),
            "fema_declarations_by_quarter": int(len(fema_declarations_by_quarter)),
            "sem_coverage_report": int(len(sem_coverage_report)),
            "sem_variable_dictionary": int(len(sem_variable_dictionary)),
        },
    }
    quality = _quality_report(
        row_counts=manifest["row_counts"],
        documents=documents,
        panel_disaster_quarter=panel_disaster_quarter,
        activities_unique=activities_unique,
        money_by_quarter=money_by_quarter,
        max_quarter_drop_pct=float(max_quarter_drop_pct),
    )

    # Additional SEM-specific quality checks (coverage + usable signal density).
    if not sem_coverage_report.empty:
        def _cov_value(panel: str, column: str, field: str) -> Optional[float]:
            hit = sem_coverage_report[
                (sem_coverage_report["panel"] == panel) & (sem_coverage_report["column"] == column)
            ]
            if hit.empty:
                return None
            return float(hit.iloc[0][field])

        sem_thresholds = [
            ("panel_disaster_quarter_sem", "programs_total", 95.0),
            ("panel_county_quarter_sem", "programs_total", 90.0),
            ("panel_city_quarter_sem", "programs_total", 85.0),
            ("panel_state_quarter_sem", "programs_total", 95.0),
        ]
        for panel_name, column_name, min_pct in sem_thresholds:
            pct = _cov_value(panel_name, column_name, "pct_non_null")
            ok = pct is not None and pct >= min_pct
            quality["checks"].append(
                {
                    "type": "sem_coverage",
                    "panel": panel_name,
                    "column": column_name,
                    "pct_non_null": pct,
                    "min_pct_non_null": min_pct,
                    "ok": bool(ok),
                }
            )
            if not ok:
                quality["failures"].append(
                    f"{panel_name}.{column_name}: non-null coverage {pct} below minimum {min_pct}"
                )

        signal_cols = [
            "admin_staff_count_sum",
            "admin_payroll_usd_sum",
            "severity_deaths_count_sum",
            "severity_economic_loss_usd_sum",
            "severity_property_damage_usd_sum",
            "severity_unmet_need_usd_sum",
        ]
        non_zero_hits = 0
        for c in signal_cols:
            nz = _cov_value("panel_disaster_quarter_sem", c, "non_zero_rows")
            if nz is not None and nz > 0:
                non_zero_hits += 1
        signal_ok = non_zero_hits >= 3
        quality["checks"].append(
            {
                "type": "sem_signal_density",
                "panel": "panel_disaster_quarter_sem",
                "non_zero_signal_columns": int(non_zero_hits),
                "min_required": 3,
                "ok": bool(signal_ok),
            }
        )
        if not signal_ok:
            quality["failures"].append(
                "panel_disaster_quarter_sem: fewer than 3 SEM signal columns have non-zero values"
            )

    quality["ok"] = len(quality.get("failures", [])) == 0
    manifest["quality"] = {
        "ok": bool(quality.get("ok")),
        "failure_count": int(len(quality.get("failures", []))),
    }

    (out_dir / "meta" / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out_dir / "meta" / "quality_report.json").write_text(json.dumps(quality, indent=2))

    removed = _cleanup_macos_artifacts(out_dir)
    if removed:
        manifest["macos_artifacts_removed"] = removed
        (out_dir / "meta" / "manifest.json").write_text(json.dumps(manifest, indent=2))
        _cleanup_macos_artifacts(out_dir)

    con.close()
    if not bool(quality.get("ok")) and not bool(allow_partial):
        failures = quality.get("failures", [])
        preview = "; ".join(str(x) for x in failures[:3])
        raise RuntimeError(
            "Data quality checks failed. "
            f"{preview}. Re-run with --allow-partial to keep outputs despite failed gates."
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready datasets for EDA/statistical models")
    parser.add_argument("--db", type=str, default=str(DEFAULT_DB_PATH), help="Path to SQLite database")
    parser.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="Output directory (default: outputs/model_ready)")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Do not fail if quality gates fail (writes quality_report.json with failure details)",
    )
    parser.add_argument(
        "--max-quarter-drop-pct",
        type=float,
        default=80.0,
        help="Maximum allowed drop vs prior quarter for key metrics before failing quality gates",
    )
    args = parser.parse_args()

    manifest = build_outputs(
        Path(args.db),
        Path(args.out),
        allow_partial=bool(args.allow_partial),
        max_quarter_drop_pct=float(args.max_quarter_drop_pct),
    )
    print("Wrote model-ready datasets:")
    for name, rel in manifest.get("outputs", {}).items():
        print(f"  {name:<24} {rel}")
    print(f"Manifest: outputs/model_ready/meta/manifest.json")
    print(f"Quality report: outputs/model_ready/meta/quality_report.json")


if __name__ == "__main__":
    main()
