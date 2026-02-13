#!/usr/bin/env python3
"""
Import legacy capacity-SEM artifacts into this repository with deduplication.

Source defaults to:
  /Volumes/T9/Projects/capacity-sem-migrated/figures

Output defaults to:
  outputs/legacy/capacity_sem_migrated/
    files/      # deduplicated artifacts
    manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path("/Volumes/T9/Projects/capacity-sem-migrated/figures")
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / "legacy" / "capacity_sem_migrated"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_rank(path: Path, source_root: Path) -> Tuple[int, int, str]:
    rel = path.relative_to(source_root)
    return (len(rel.parts), len(str(rel)), str(rel))


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _cleanup_macos_sidecars(root: Path) -> int:
    removed = 0
    for path in root.rglob("._*"):
        if path.is_file():
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def import_legacy_artifacts(
    source_root: Path,
    output_root: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, object]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source is not a directory: {source_root}")

    source_files = list(_iter_files(source_root))
    grouped_by_hash: Dict[str, List[Path]] = {}
    for src in source_files:
        h = _hash_file(src)
        grouped_by_hash.setdefault(h, []).append(src)

    files_out_root = output_root / "files"
    manifest_entries: List[Dict[str, object]] = []
    duplicates = 0
    redundant = 0

    for h in sorted(grouped_by_hash.keys()):
        candidates = grouped_by_hash[h]
        if len(candidates) > 1:
            duplicates += 1
            redundant += len(candidates) - 1
        canonical = sorted(candidates, key=lambda p: _canonical_rank(p, source_root))[0]
        rel = canonical.relative_to(source_root)
        dest = files_out_root / rel
        variants = [str(p.relative_to(source_root)) for p in sorted(candidates)]

        manifest_entries.append(
            {
                "sha256": h,
                "size_bytes": int(canonical.stat().st_size),
                "destination": str(Path("files") / rel),
                "canonical_source": str(rel),
                "source_variants": variants,
                "variant_count": len(candidates),
            }
        )

        if dry_run:
            continue

        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            existing_hash = _hash_file(dest)
            if existing_hash == h:
                continue
            if not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite non-matching file: {dest} "
                    "(rerun with --overwrite to allow)"
                )
        # copyfile avoids preserving metadata that can create AppleDouble sidecars
        # on some external filesystems.
        shutil.copyfile(canonical, dest)

    manifest = {
        "built_at_utc": _utc_now(),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "dry_run": bool(dry_run),
        "source_file_count": len(source_files),
        "unique_file_count": len(grouped_by_hash),
        "duplicate_group_count": duplicates,
        "redundant_source_file_count": redundant,
        "entries": manifest_entries,
    }

    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        removed_sidecars = _cleanup_macos_sidecars(output_root)
        manifest["macos_sidecars_removed"] = int(removed_sidecars)
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and import legacy capacity-SEM artifacts")
    parser.add_argument(
        "--source",
        type=str,
        default=str(DEFAULT_SOURCE_ROOT),
        help="Source directory (default: /Volumes/T9/Projects/capacity-sem-migrated/figures)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Output directory (default: outputs/legacy/capacity_sem_migrated)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting non-matching existing destination files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute manifest only; do not copy files",
    )
    args = parser.parse_args()

    manifest = import_legacy_artifacts(
        source_root=Path(args.source),
        output_root=Path(args.out),
        overwrite=bool(args.overwrite),
        dry_run=bool(args.dry_run),
    )

    print("Legacy import summary:")
    print(f"  Source files: {manifest['source_file_count']}")
    print(f"  Unique files copied: {manifest['unique_file_count']}")
    print(f"  Duplicate groups: {manifest['duplicate_group_count']}")
    print(f"  Redundant source files: {manifest['redundant_source_file_count']}")
    if bool(args.dry_run):
        print("  Dry run: no files copied")
    else:
        print(f"  Manifest: {Path(args.out) / 'manifest.json'}")


if __name__ == "__main__":
    main()
