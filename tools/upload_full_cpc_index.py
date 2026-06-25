#!/usr/bin/env python3
"""
upload_full_cpc_index.py  --  Publish the FULL CPC exemplar index as a PRIVATE
HuggingFace dataset.

Companion to ``build_literature_index/03_upload_huggingface.py``, but for the
presentation-of-case exemplar index rather than the literature index, and PRIVATE
by default (the full >6,000-case CPC corpus cannot be redistributed publicly).

Input is the single parquet produced by the main repo's
``benchmarks/processing/build_full_cpc_presentation_index.py``
(``cpc_presentation_index_full.parquet``) — same schema as the public 100-case
parquet, so CaBot-Public loads it unchanged.

An authenticated user with read access to the repo can then pull it with
``fetch_data.py --full-cpc-index`` (which calls snapshot_download with the HF
token); without access, CaBot-Public falls back to the public 100-case index.

Uses the HF write token from config.ini ([main] HF_WRITE_TOKEN) or env HF_TOKEN.

Usage:
    python tools/upload_full_cpc_index.py
    python tools/upload_full_cpc_index.py --repo-id tbuckley/cabot-cpc-index-full
    python tools/upload_full_cpc_index.py --parquet data/cpc_presentation_index_full.parquet
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent / "build_literature_index"))
from common import EMBED_DIM, EMBED_MODEL, get_hf_token  # noqa: E402

DEFAULT_REPO = "tbuckley/cabot-cpc-index-full"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PARQUET = REPO_ROOT / "data" / "cpc_presentation_index_full.parquet"
PATH_IN_REPO = "cpc_presentation_index_full.parquet"

# Dataset card written to the repo root so the (private) HF page documents the schema.
DATASET_CARD = f"""---
license: other
extra_gated_prompt: >-
  The full NEJM CPC exemplar corpus cannot be redistributed. Access is restricted.
tags:
- clinical
- biomedical
- nejm
- cpc
- embeddings
pretty_name: CaBot Full CPC Exemplar Index (private)
configs:
- config_name: default
  data_files:
  - split: train
    path: cpc_presentation_index_full.parquet
---

# CaBot Full CPC Exemplar Index (private)

The full presentation-of-case exemplar-retrieval index used by **CaBot v1** (the
physician A/B-test model). Each row is one NEJM Clinicopathologic Conference (CPC)
case: the presentation-of-case text, its precomputed `{EMBED_MODEL}` embedding
({EMBED_DIM}-d, float32), the differential-diagnosis section, and id / title / year /
decade / publication_date metadata.

Corpus span: publication dates **1923-10-25 .. 2024-05-30** (cases on/before the o3
pretraining cutoff). This is the private counterpart to the public 100-case index
shipped with CaBot-Public; it is **not** publicly redistributable, hence this private
dataset.

## Columns

| column | type | notes |
|---|---|---|
| id | string | NEJM case id (used for `--exclude-id` filtering) |
| title | string | |
| publication_date | string | YYYY-MM-DD |
| year | int64 | used for year_min / year_max retrieval filters |
| decade | int64 | used for the decade filter |
| presentation_of_case | string | the embedded document (head/tail truncated past 8000 tokens) |
| differential_diagnosis | string | exemplar DDx returned to the model |
| token_count | int64 | tokens in presentation_of_case |
| embedding_model | string | `{EMBED_MODEL}` |
| embedding_input_truncated | bool | whether the presentation was truncated before embedding |
| embedding | list[float32] x {EMBED_DIM} | `{EMBED_MODEL}` over presentation_of_case |

## Use

CaBot-Public loads this with the same `CPCPresentationStore` as the public index:
similarity is cosine (L2-normalized inner product). Pull it with
`fetch_data.py --full-cpc-index` (requires an HF token with access).
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default=DEFAULT_REPO,
                    help=f"HuggingFace dataset repo (default: {DEFAULT_REPO})")
    ap.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET,
                    help=f"Full CPC index parquet to upload (default: {DEFAULT_PARQUET})")
    ap.add_argument("--public", action="store_true",
                    help="Create the repo PUBLIC (default: private). Not recommended — the "
                         "full CPC corpus is not redistributable.")
    ap.add_argument("--gated", choices=["auto", "manual"], default=None,
                    help="Enable HF gated access (request-and-approve) on top of visibility. "
                         "'auto' grants on request; 'manual' requires owner approval.")
    args = ap.parse_args()

    if not args.parquet.exists():
        ap.error(f"Parquet not found: {args.parquet}. Build it first with "
                 f"benchmarks/processing/build_full_cpc_presentation_index.py")

    private = not args.public
    size_gb = args.parquet.stat().st_size / 1e9
    api = HfApi(token=get_hf_token())

    print(f"Creating/looking up dataset repo: {args.repo_id} ({'private' if private else 'PUBLIC'})")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=private, exist_ok=True)
    # Enforce intended visibility even if the repo pre-existed. `private` moved from
    # update_repo_visibility (older hub) to update_repo_settings (newer hub), so try both.
    try:
        api.update_repo_settings(repo_id=args.repo_id, repo_type="dataset", private=private)
    except TypeError:
        api.update_repo_visibility(repo_id=args.repo_id, repo_type="dataset", private=private)
    if args.gated:
        api.update_repo_settings(repo_id=args.repo_id, repo_type="dataset", gated=args.gated)
        print(f"Gated access set to: {args.gated}")

    # 1) Dataset card at repo root (its `configs:` pins the train split to the parquet).
    api.upload_file(
        repo_id=args.repo_id, repo_type="dataset", path_in_repo="README.md",
        path_or_fileobj=DATASET_CARD.encode(),
    )
    # 2) The single parquet (LFS), at the path CaBot-Public expects.
    print(f"Uploading {args.parquet.name} ({size_gb:.2f} GB) ...")
    api.upload_file(
        repo_id=args.repo_id, repo_type="dataset", path_in_repo=PATH_IN_REPO,
        path_or_fileobj=str(args.parquet),
    )

    print(f"\nDone. https://huggingface.co/datasets/{args.repo_id}")
    print("Pull it into CaBot-Public with: python fetch_data.py --full-cpc-index")


if __name__ == "__main__":
    main()
