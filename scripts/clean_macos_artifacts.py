#!/usr/bin/env python3
"""
Remove macOS Finder/AppleDouble artifacts from the working tree.

These files commonly appear on external drives and confuse non-technical users:
- .DS_Store
- ._*

This script avoids touching `.git/` and common virtualenv folders.
"""

from __future__ import annotations

import argparse
from pathlib import Path


EXCLUDE_DIR_NAMES = {".git", "venv", ".venv", "node_modules", "__pycache__"}


def should_skip_dir(path: Path) -> bool:
    return path.name in EXCLUDE_DIR_NAMES


def clean(root: Path) -> int:
    removed = 0
    for path in root.rglob("*"):
        if path.is_dir() and should_skip_dir(path):
            # rglob will still descend; we rely on checks below for safety
            continue
        if not path.is_file():
            continue
        if path.name == ".DS_Store" or path.name.startswith("._"):
            # Never touch .git contents
            if any(p.name == ".git" for p in path.parents):
                continue
            if any(p.name in {"venv", ".venv", "node_modules"} for p in path.parents):
                continue
            try:
                path.unlink()
                removed += 1
            except OSError:
                continue
    return removed


def main() -> None:
    parser = argparse.ArgumentParser(description="Delete macOS AppleDouble/.DS_Store artifacts")
    parser.add_argument("--root", type=str, default=".", help="Root directory to clean (default: .)")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    removed = clean(root)
    print(f"Removed {removed} artifacts under {root}")


if __name__ == "__main__":
    main()

