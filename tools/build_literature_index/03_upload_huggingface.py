#!/usr/bin/env python3
"""
03_upload_huggingface.py  --  Publish the parquet shards as a HuggingFace dataset.

THIRD stage. Uploads the sharded float32 parquet produced by stage 2 to a
HuggingFace dataset repo, so it can be downloaded with one line (see 04_load_postgres.py)
and linked from a website. Uses the HF write token from config.ini ([main]
HF_WRITE_TOKEN) or the HF_TOKEN env var.

The dataset is large (~25-30 GB across shards). We use upload_large_folder, which
is resumable and chunk-uploads each shard over Git-LFS, so an interrupted upload
can be re-run safely.

Usage:
    python 03_upload_huggingface.py --repo-id your-username/cabot-search-index
    python 03_upload_huggingface.py --repo-id ... --private
    python 03_upload_huggingface.py --repo-id ... --local-dir data/parquet
"""

import argparse
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, EMBED_DIM, EMBED_MODEL, JOURNALS_DIR, get_hf_token  # noqa: E402

DEFAULT_LOCAL_DIR = DATA_DIR / "parquet"

# Dataset card written to the repo root so the HF page documents the schema.
DATASET_CARD = f"""---
license: cc0-1.0
task_categories:
- feature-extraction
- sentence-similarity
tags:
- clinical
- biomedical
- openalex
- embeddings
pretty_name: CaBot Clinical Literature Embedding Index
configs:
- config_name: default
  data_files:
  - split: train
    path: part-*.parquet
---

# CaBot Clinical Literature Embedding Index

Exact embedding-search index over 3,474,244 works from 204 high-impact clinical
journals (2023 JIF >= 10), built from an OpenAlex snapshot (~June 2025) and
embedded with OpenAI `{EMBED_MODEL}` at {EMBED_DIM} dimensions (float32).

This is used for CaBot. The build and search code is in
`tools/build_literature_index/` of the CaBot source repository.

## Columns

| column | type | notes |
|---|---|---|
| id | string | OpenAlex work id |
| doi | string | |
| title | string | |
| abstract | string | reconstructed from OpenAlex inverted index ("" if none) |
| journal | string | |
| year | int32 | |
| publication_date | string | YYYY-MM-DD |
| cited_by_count | int32 | |
| authors | list[string] | raw author names |
| is_pubmed_indexed | bool | |
| is_open_access | bool | |
| article_type | string | OpenAlex `type` |
| has_abstract | bool | |
| embedding | list[float32] x {EMBED_DIM} | `{EMBED_MODEL}`, document text = lowercased title (+ abstract) |

## Reproducing search

```python
from datasets import load_dataset
ds = load_dataset("REPO_ID", split="train")   # streams the parquet shards
```

Embed queries with a `"query: "` prefix and use cosine similarity. To reproduce
the production engine exactly, load the shards into PostgreSQL + pgvector with
`04_load_postgres.py` (IVFFlat, lists=1732, probes=42) and search with
`05_search.py`.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", required=True,
                    help="HuggingFace dataset repo, e.g. your-username/cabot-search-index")
    ap.add_argument("--local-dir", type=Path, default=DEFAULT_LOCAL_DIR,
                    help=f"Local parquet dir to upload (default: {DEFAULT_LOCAL_DIR})")
    ap.add_argument("--private", action="store_true", help="Create the repo as private.")
    args = ap.parse_args()

    if not args.local_dir.exists():
        ap.error(f"Local dir not found: {args.local_dir}. Run 02_build_embeddings.py first.")
    shards = sorted(args.local_dir.glob("part-*.parquet"))
    if not shards:
        ap.error(f"No part-*.parquet shards in {args.local_dir}.")

    total_gb = sum(p.stat().st_size for p in shards) / 1e9
    api = HfApi(token=get_hf_token())
    print(f"Creating/looking up dataset repo: {args.repo_id}")
    api.create_repo(repo_id=args.repo_id, repo_type="dataset",
                    private=args.private, exist_ok=True)
    # Enforce intended visibility even if the repo pre-existed or the namespace
    # defaults new repos to private (so a public dataset is downloadable with no token).
    api.update_repo_settings(repo_id=args.repo_id, repo_type="dataset",
                             private=args.private)

    # 1) Dataset card at repo root. Its `configs:` pins the train split to
    #    data/*.parquet, so load_dataset() is unambiguous.
    api.upload_file(
        repo_id=args.repo_id, repo_type="dataset", path_in_repo="README.md",
        path_or_fileobj=DATASET_CARD.replace("REPO_ID", args.repo_id).encode(),
    )
    # 2) Journal selection files (provenance) under journals/.
    api.upload_folder(
        repo_id=args.repo_id, repo_type="dataset",
        folder_path=str(JOURNALS_DIR), path_in_repo="journals",
    )
    # 3) The parquet shards at the repo root (the card's `configs:` pins the
    #    train split to part-*.parquet). upload_large_folder is resumable and
    #    chunk-uploads each LFS file, so an interrupted run can be re-run safely.
    print(f"Uploading {len(shards)} parquet shards ({total_gb:.1f} GB) ...")
    api.upload_large_folder(
        repo_id=args.repo_id, repo_type="dataset",
        folder_path=str(args.local_dir),
        allow_patterns=["part-*.parquet"],
    )

    print(f"\nDone. https://huggingface.co/datasets/{args.repo_id}")
    print("Update 04_load_postgres.py / the website download link to point at this repo.")


if __name__ == "__main__":
    main()
