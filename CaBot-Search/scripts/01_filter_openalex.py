#!/usr/bin/env python3
"""
01_filter_openalex.py  --  Raw OpenAlex snapshot -> high-impact clinical subset.

This is the FIRST stage of reproducing the CaBot search index.
It walks a local OpenAlex data snapshot and keeps only works published in our
204 high-impact clinical journals (2023 Journal Impact Factor >= 10, Clarivate
JCR), writing them to a single compressed JSONL file.

------------------------------------------------------------------------------
PROVENANCE / REPRODUCIBILITY
------------------------------------------------------------------------------
The hosted index was originally built in **early June 2025** (OpenAlex snapshot
pulled ~2025-06-05, subset extracted 2025-06-06). OpenAlex is a living database:
works get added, citation counts change, and journal<->source-id assignments can
shift over time. If you run this against a newer snapshot you will get a *similar*
but not identical subset to the one described in the paper. To reproduce the exact
index, use an OpenAlex snapshot from on/around 2025-06-05.

How to get an OpenAlex snapshot (free, no AWS account needed):
    aws s3 sync "s3://openalex" "openalex-snapshot" --no-sign-request
We only need the works/ tree:
    aws s3 sync "s3://openalex/data/works" "openalex-snapshot/data/works" --no-sign-request
(The sources/ tree was used once to build journals/journal_source_ids.json; that
file is already checked in, so you do not need sources/ to reproduce the subset.)

------------------------------------------------------------------------------
JOURNAL SELECTION
------------------------------------------------------------------------------
The journal universe is the 204 titles in journals/high_impact_journals_if10.txt
(every clinical/biomedical journal with 2023 JIF >= 10 in the JCR export
journals/top_journals_jcr_2023.txt). Each title was matched to its OpenAlex
source id(s); journals/journal_source_ids.json holds that mapping. A work is kept
if ANY of its locations (primary or secondary) points at one of those source ids.

Usage:
    python 01_filter_openalex.py --snapshot /path/to/openalex-snapshot/data/works
    python 01_filter_openalex.py --snapshot ... --limit 100000   # quick test run
"""

import argparse
import gzip
import sys
from pathlib import Path

import orjson
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import JOURNALS_DIR, DATA_DIR, reconstruct_abstract  # noqa: E402

SOURCE_IDS_FILE = JOURNALS_DIR / "journal_source_ids.json"
DEFAULT_OUT = DATA_DIR / "journal_works.jsonl.gz"


def load_target_source_ids():
    """Load the set of OpenAlex source ids we want to keep, plus a reverse map
    source_id -> journal name (for the end-of-run breakdown)."""
    data = orjson.loads(SOURCE_IDS_FILE.read_bytes())
    target = set()
    source_to_journal = {}
    for journal_name, entry in data.get("matched", {}).items():
        for source_id in entry.get("source_ids", []):
            target.add(source_id)
            source_to_journal[source_id] = journal_name
    print(
        f"Loaded {len(target)} source ids across {len(data.get('matched', {}))} journals "
        f"({len(data.get('unmatched', []))} unmatched)."
    )
    return target, source_to_journal


def work_matches(work, target_source_ids):
    """Return True if any of the work's locations belongs to a target journal."""
    primary = work.get("primary_location") or {}
    primary_src = (primary.get("source") or {}).get("id")
    if primary_src in target_source_ids:
        return True
    for loc in work.get("locations", []) or []:
        src = (loc.get("source") or {}).get("id")
        if src in target_source_ids:
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", type=Path, required=True,
                    help="Path to the OpenAlex works tree, e.g. "
                         "openalex-snapshot/data/works")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Output JSONL.gz path (default: {DEFAULT_OUT})")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after scanning this many works (for quick tests).")
    args = ap.parse_args()

    if not args.snapshot.exists():
        ap.error(f"Snapshot works folder not found: {args.snapshot}")

    target_source_ids, source_to_journal = load_target_source_ids()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    gz_files = sorted(args.snapshot.rglob("*.gz"))
    print(f"Scanning {len(gz_files)} OpenAlex part files -> {args.out}")

    scanned = matched = 0
    per_journal = {}

    with gzip.open(args.out, "wb") as out:
        for gz in tqdm(gz_files, desc="part files"):
            with gzip.open(gz, "rb") as f:
                for line in f:
                    scanned += 1
                    if args.limit and scanned > args.limit:
                        break
                    try:
                        work = orjson.loads(line)
                    except orjson.JSONDecodeError:
                        continue
                    if not work_matches(work, target_source_ids):
                        continue

                    # Reconstruct the abstract once, here, so downstream stages
                    # never have to deal with OpenAlex's inverted-index format.
                    work["abstract"] = reconstruct_abstract(
                        work.get("abstract_inverted_index"))

                    out.write(orjson.dumps(work) + b"\n")
                    matched += 1

                    src = ((work.get("primary_location") or {}).get("source") or {}).get("id")
                    j = source_to_journal.get(src)
                    if j:
                        per_journal[j] = per_journal.get(j, 0) + 1
            if args.limit and scanned > args.limit:
                break

    print(f"\nDone. Scanned {scanned:,} works, kept {matched:,} "
          f"({matched / max(scanned, 1) * 100:.2f}%).")
    print(f"Wrote {args.out}")
    print("\nTop journals by work count:")
    for journal, count in sorted(per_journal.items(), key=lambda kv: kv[1], reverse=True)[:10]:
        print(f"  {count:>8,}  {journal}")


if __name__ == "__main__":
    main()
