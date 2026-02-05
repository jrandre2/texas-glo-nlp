#!/usr/bin/env python3
"""
Build a shareable, non-technical bundle of the key outputs.

The bundle is based on what is linked in TEAM_PORTAL.html, copying those files
into a new folder while preserving relative paths so the portal continues to work.

Default behavior:
  - Copies TEAM_PORTAL.html + all linked local artifacts <= --max-mb
  - Skips spatial HTML maps by default (they are often 100MB+)

Writes:
  outputs/share_bundle/        (default)
  outputs/share_bundle.zip     (optional, --zip)
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import List, Optional, Set, Tuple


ROOT = Path(__file__).resolve().parents[1]
PORTAL = ROOT / "TEAM_PORTAL.html"
DEFAULT_OUT_DIR = ROOT / "outputs" / "share_bundle"


class _HrefParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        for k, v in attrs:
            if k.lower() == "href" and v:
                self.hrefs.append(v)


class _ResourceParser(HTMLParser):
    """
    Extract local-ish resource references from common HTML tags.

    This intentionally stays simple: it looks for `src=` and `href=` attributes
    on a small set of tags so we can pull in report images (and similar assets)
    into the share bundle.
    """

    def __init__(self) -> None:
        super().__init__()
        self.refs: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        attr_map = {k.lower(): v for k, v in attrs if k}
        if t == "img":
            v = attr_map.get("src")
        elif t == "script":
            v = attr_map.get("src")
        elif t == "link":
            v = attr_map.get("href")
        else:
            v = None
        if v:
            self.refs.append(v)


def _cleanup_macos_artifacts(root: Path) -> int:
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


def _is_local_relpath(href: str) -> bool:
    h = (href or "").strip()
    if not h:
        return False
    if h.startswith("#"):
        return False
    if h.startswith(("http://", "https://", "mailto:", "data:", "javascript:")):
        return False
    # Portal only uses repo-relative paths; avoid absolute paths.
    return not h.startswith("/")


def _should_skip(relpath: str, include_spatial: bool) -> bool:
    p = relpath.replace("\\", "/")
    if p == "outputs/exports/entities.csv":
        return True
    if p.startswith("DRGR_Reports/"):
        return True
    if not include_spatial and p.startswith("outputs/exports/spatial_") and p.endswith(".html"):
        return True
    if not include_spatial and p.endswith("_joined.geojson"):
        return True
    return False


_CARD_RE = re.compile(
    r"""
    <div\s+class="card">\s*
      <div\s+class="card-title"><a\s+href="(?P<href>[^"]+)">.*?</a></div>\s*
      <div\s+class="card-meta">.*?</div>\s*
      <div\s+class="card-desc">.*?</div>\s*
    </div>
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

_SECTION_RE = re.compile(
    r"""
    <section>\s*
      <h2>(?P<title>.*?)</h2>\s*
      (?P<body>.*?)\s*
    </section>
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)


def _filter_portal_for_bundle(html_text: str, bundle_root: Path) -> str:
    """
    Remove cards that link to files not present in the bundle.
    If a section ends up empty, insert a small muted message.
    """

    def card_repl(match: re.Match[str]) -> str:
        href = (match.group("href") or "").strip()
        rel = href.split("?", 1)[0].split("#", 1)[0].strip()
        if _is_local_relpath(rel) and not (bundle_root / rel).exists():
            return ""
        return match.group(0)

    filtered = _CARD_RE.sub(card_repl, html_text)

    def section_repl(match: re.Match[str]) -> str:
        body = match.group("body") or ""
        if "class=\"card\"" in body or "class='card'" in body:
            return match.group(0)
        if "class=\"muted\"" in body or "class='muted'" in body:
            return match.group(0)
        title = match.group("title") or ""
        return (
            "<section>\n"
            f"  <h2>{title}</h2>\n"
            "  <div class=\"muted\">Not included in this share bundle. See README.md.</div>\n"
            "</section>"
        )

    return _SECTION_RE.sub(section_repl, filtered)


def _extract_local_resource_relpaths(html_path: Path) -> Set[str]:
    """
    Return repo-relative paths for local resources referenced by an HTML file.

    Example:
      outputs/reports/foo.html referencing `assets/chart.png` yields
      outputs/reports/assets/chart.png
    """

    try:
        html_text = html_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        html_text = html_path.read_text(encoding="utf-8", errors="ignore")

    parser = _ResourceParser()
    parser.feed(html_text)

    relpaths: Set[str] = set()
    for ref in parser.refs:
        if not _is_local_relpath(ref):
            continue
        rel = ref.split("?", 1)[0].split("#", 1)[0].strip()
        if not rel:
            continue
        abs_path = (html_path.parent / rel).resolve()
        try:
            root_rel = abs_path.relative_to(ROOT)
        except ValueError:
            continue
        relpaths.add(root_rel.as_posix())

    return relpaths


def build_bundle(out_dir: Path, max_mb: float, include_spatial: bool, make_zip: bool) -> None:
    if not PORTAL.exists():
        raise FileNotFoundError(f"Missing portal: {PORTAL}")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    portal_html = PORTAL.read_text(encoding="utf-8")

    parser = _HrefParser()
    parser.feed(portal_html)

    max_bytes = int(max_mb * 1024 * 1024)
    copied: Set[str] = set()
    skipped: List[str] = []

    def maybe_copy(rel: str) -> None:
        rel = (rel or "").strip()
        if not rel or rel in copied:
            return
        if _should_skip(rel, include_spatial=include_spatial):
            skipped.append(rel)
            return

        src = ROOT / rel
        if not src.exists() or not src.is_file():
            return

        try:
            size = src.stat().st_size
        except OSError:
            return
        if size > max_bytes:
            skipped.append(rel)
            return

        dst = out_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied.add(rel)

    for href in parser.hrefs:
        if not _is_local_relpath(href):
            continue
        rel = href.split("?", 1)[0].split("#", 1)[0].strip()
        maybe_copy(rel)

    # Copy dependencies referenced by included HTML artifacts (e.g., report images in assets/)
    deps: Set[str] = set()
    for rel in list(copied):
        if not rel.lower().endswith(".html"):
            continue
        src_html = ROOT / rel
        if not src_html.exists() or not src_html.is_file():
            continue
        deps.update(_extract_local_resource_relpaths(src_html))
    for dep in sorted(deps):
        maybe_copy(dep)

    # Write a filtered portal that only links to included artifacts
    (out_dir / "TEAM_PORTAL.html").write_text(_filter_portal_for_bundle(portal_html, out_dir), encoding="utf-8")

    # Add a tiny bundle README for email/drive handoff
    readme = out_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Texas GLO DRGR – Share Bundle",
                "",
                "Open `TEAM_PORTAL.html` to browse the included outputs.",
                "",
                "Notes:",
                f"- Built from: `{ROOT}`",
                f"- Max file size copied: {max_mb:.1f} MB",
                "- Spatial maps are excluded by default (they are often very large).",
                "",
            ]
        ),
        encoding="utf-8",
    )

    _cleanup_macos_artifacts(out_dir)

    if make_zip:
        zip_path = out_dir.with_suffix(".zip")
        if zip_path.exists():
            zip_path.unlink()
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for path in out_dir.rglob("*"):
                if path.is_file():
                    zf.write(path, path.relative_to(out_dir))

        _cleanup_macos_artifacts(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a shareable bundle from TEAM_PORTAL-linked artifacts")
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT_DIR), help="Output folder (default: outputs/share_bundle)")
    ap.add_argument("--max-mb", type=float, default=50.0, help="Max file size to include (default: 50MB)")
    ap.add_argument("--include-spatial", action="store_true", help="Include spatial HTML maps / joined GeoJSONs (can be huge)")
    ap.add_argument("--zip", action="store_true", help="Also create outputs/share_bundle.zip")
    args = ap.parse_args()

    build_bundle(Path(args.out), max_mb=float(args.max_mb), include_spatial=bool(args.include_spatial), make_zip=bool(args.zip))
    print(f"Wrote bundle: {args.out}")
    if args.zip:
        print(f"ZIP: {Path(args.out).with_suffix('.zip')}")


if __name__ == "__main__":
    main()
