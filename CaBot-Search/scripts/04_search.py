#!/usr/bin/env python3
"""
04_search.py  --  Load the hosted index and run exact embedding search locally.

This is the QUICKSTART entry point. It loads the embeddings + metadata (from a
local parquet dir, or downloaded from HuggingFace) and searches them with exact
cosine similarity (the query is embedded the same way CaBot does: "query: " +
text). Two output modes:

  * default  -- a compact human-readable ranking (uses FAISS IndexFlatIP).
  * --json   -- the EXACT CaBot /api/search JSON response, byte-for-byte
                (same fields, formatting, dual abstract/title retrieval and
                yearFrom/yearTo/citationsMin/citationsMax/journals filters).

vs. the production API: results are computed by exact brute-force search here
(identical or better recall) rather than the API's approximate IVFFlat index, and
the float `score` can differ in its trailing digits (pgvector vs. numpy
arithmetic). Everything else in --json mode is byte-identical to the API.

Similarity = cosine: vectors are L2-normalized and compared by inner product, so
score == cosine similarity == the API's 1 - cosine_distance.

------------------------------------------------------------------------------
MEMORY: the full index is 3.47M x 1536 float32 ~= 21 GB in RAM (FAISS holds a
second copy in default mode -> ~45 GB; --json mode skips FAISS). Use --max-rows
to load a subset on a smaller machine (results are then drawn from that subset).
------------------------------------------------------------------------------

Usage:
    python 04_search.py                                   # samples, hosted index
    python 04_search.py --query "GLP-1 cardiovascular outcomes" --k 10
    # Exact CaBot API response as JSON:
    python 04_search.py --query "CAR-T relapsed B-cell lymphoma" --json
    python 04_search.py --query "statins" --json --need-abstract \\
        --year-from 2015 --year-to 2023 --citations-min 50
    python 04_search.py --local data/parquet --json --query "sepsis bundles"
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apiformat  # noqa: E402
from common import EMBED_DIM, build_query_text, get_embedding_client  # noqa: E402

SAMPLE_QUERIES = [
    "GLP-1 receptor agonists and cardiovascular outcomes in type 2 diabetes",
    "CAR-T cell therapy for relapsed large B-cell lymphoma",
    "mRNA vaccine efficacy against severe COVID-19",
]

DEFAULT_REPO = "tbuckley/cabot-search"  # the hosted index; used if no source given

# Columns loaded for each mode (besides the embedding vector).
HUMAN_COLS = ["id", "doi", "title", "journal", "year", "cited_by_count",
              "has_abstract", "is_open_access"]
# Everything the /api/search response needs (matches apiformat.build_result).
API_COLS = ["id", "doi", "title", "abstract", "journal", "year", "publication_date",
            "cited_by_count", "authors", "is_pubmed_indexed", "is_open_access",
            "article_type", "has_abstract", "biblio", "oa_locations"]


def resolve_parquet_dir(args):
    """Return a local directory of parquet shards, downloading from HF if needed."""
    if args.local:
        d = Path(args.local)
        if not d.exists():
            sys.exit(f"Local dir not found: {d}")
        return d
    # Download (and cache) the hosted dataset's parquet shards from HuggingFace.
    # A token is used if one is configured (needed for private repos / better rate
    # limits) but a public dataset downloads fine anonymously.
    from huggingface_hub import snapshot_download
    try:
        from common import get_hf_token
        token = get_hf_token()
    except Exception:
        token = None
    log(f"Downloading {args.repo_id} from HuggingFace (cached after first run)...")
    path = snapshot_download(repo_id=args.repo_id, repo_type="dataset",
                             allow_patterns=["*.parquet"], token=token)
    return Path(path)


def normalize_rows(x):
    """In-place L2 normalize rows (so inner product == cosine similarity)."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    np.divide(x, norms, out=x, where=norms > 0)


def load_index(parquet_dir, cols, max_rows=None):
    """Load the embedding matrix (float32, L2-normalized) + metadata `cols`."""
    dataset = ds.dataset(str(parquet_dir), format="parquet")
    log(f"Loading vectors + metadata from {parquet_dir} ...")
    vecs, meta = [], {c: [] for c in cols}
    loaded = 0
    for batch in dataset.to_batches(columns=cols + ["embedding"]):
        # embedding is fixed_size_list<float>[1536]; flatten child values + reshape.
        flat = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
        vecs.append(flat.reshape(-1, EMBED_DIM).astype(np.float32, copy=False))
        for c in cols:
            meta[c].extend(batch.column(c).to_pylist())
        loaded += len(batch)
        if max_rows and loaded >= max_rows:
            break
    embeddings = np.vstack(vecs)
    if max_rows:
        embeddings = embeddings[:max_rows]
        meta = {c: v[:max_rows] for c, v in meta.items()}
    normalize_rows(embeddings)
    log(f"Loaded {embeddings.shape[0]:,} vectors of dim {embeddings.shape[1]}.")
    return embeddings, meta


def embed_query(client, model_id, query):
    """Embed a query as production does: prefix 'query: ', then L2-normalize."""
    resp = client.embeddings.create(
        model=model_id, input=build_query_text(query), dimensions=EMBED_DIM)
    v = np.asarray([resp.data[0].embedding], dtype=np.float32)
    normalize_rows(v)
    return v[0]


# --------------------------------------------------------------------------- #
#  --json mode: reproduce the CaBot /api/search response exactly
# --------------------------------------------------------------------------- #
def filter_mask(meta, n, args):
    """Boolean mask over all rows for the year / citations / journals filters,
    matching the API's SQL WHERE clause (NULL year excluded when a year filter is
    set; citations use COALESCE(...,0))."""
    mask = np.ones(n, dtype=bool)
    has_year = np.array([y is not None for y in meta["year"]])
    years = np.array([y if y is not None else 0 for y in meta["year"]], dtype=np.int64)
    if args.year_from and args.year_to:
        mask &= has_year & (years >= int(args.year_from)) & (years <= int(args.year_to))
    elif args.year_from:
        mask &= has_year & (years >= int(args.year_from))
    elif args.year_to:
        mask &= has_year & (years <= int(args.year_to))

    cites = np.array([c if c is not None else 0 for c in meta["cited_by_count"]],
                     dtype=np.int64)
    if args.citations_min and args.citations_max:
        mask &= (cites >= int(args.citations_min)) & (cites <= int(args.citations_max))
    elif args.citations_min:
        mask &= cites >= int(args.citations_min)
    elif args.citations_max:
        mask &= cites <= int(args.citations_max)

    if args.journals:
        wanted = set(args.journals.split(","))
        mask &= np.array([j in wanted for j in meta["journal"]])
    return mask


def top5(scores, eligible):
    """Indices of the top-5 by score among rows where `eligible` is True (desc)."""
    idxs = np.nonzero(eligible)[0]
    if idxs.size == 0:
        return []
    sub = scores[idxs]
    k = min(5, idxs.size)
    top = np.argpartition(-sub, k - 1)[:k]
    top = top[np.argsort(-sub[top])]
    return idxs[top].tolist()


def row_for_output(meta, i, score):
    """Assemble the raw row dict apiformat.build_result expects."""
    return {c: meta[c][i] for c in API_COLS} | {"score": float(score)}


def search_api(embeddings, meta, qvec, args):
    """Run the dual abstract/title retrieval + filters and return ordered indices."""
    scores = embeddings @ qvec  # cosine similarity (both sides unit-normalized)
    base = filter_mask(meta, embeddings.shape[0], args)
    has_abs = np.array(meta["has_abstract"], dtype=bool)
    if args.need_abstract:
        order = top5(scores, base & has_abs)
    else:
        # API combines: no-abstract results first, then with-abstract.
        order = top5(scores, base & ~has_abs) + top5(scores, base & has_abs)
    return [row_for_output(meta, i, scores[i]) for i in order]


def log(msg):
    """Progress/info goes to stderr so --json output on stdout stays clean."""
    print(msg, file=sys.stderr)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--local", help="Local parquet dir (from 02_build_embeddings.py).")
    src.add_argument("--repo-id", help=f"HuggingFace dataset repo (default: {DEFAULT_REPO}).")
    ap.add_argument("--query", help="A single query (default: built-in samples).")
    ap.add_argument("--k", type=int, default=5, help="Results per query, human mode (default: 5).")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Load only this many vectors (for low-RAM machines).")
    # --json mode: exact CaBot /api/search response + its filters.
    ap.add_argument("--json", action="store_true",
                    help="Emit the exact CaBot /api/search JSON response on stdout.")
    ap.add_argument("--need-abstract", action="store_true",
                    help="[--json] needAbstract=true (only papers with abstracts).")
    ap.add_argument("--year-from", help="[--json] yearFrom filter.")
    ap.add_argument("--year-to", help="[--json] yearTo filter.")
    ap.add_argument("--citations-min", help="[--json] citationsMin filter.")
    ap.add_argument("--citations-max", help="[--json] citationsMax filter.")
    ap.add_argument("--journals", help="[--json] comma-separated journal names filter.")
    args = ap.parse_args()

    if not args.local and not args.repo_id:
        args.repo_id = DEFAULT_REPO

    parquet_dir = resolve_parquet_dir(args)
    client, model_id = get_embedding_client()

    if args.json:
        embeddings, meta = load_index(parquet_dir, API_COLS, args.max_rows)
        query = args.query or SAMPLE_QUERIES[0]
        qvec = embed_query(client, model_id, query)
        rows = search_api(embeddings, meta, qvec, args)
        resp = apiformat.build_response(query, rows, user_api_key_present=True)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return

    # Human-readable mode (FAISS exact search).
    import faiss
    embeddings, meta = load_index(parquet_dir, HUMAN_COLS, args.max_rows)
    index = faiss.IndexFlatIP(EMBED_DIM)
    index.add(embeddings)
    queries = [args.query] if args.query else SAMPLE_QUERIES
    for query in queries:
        qv = embed_query(client, model_id, query).reshape(1, -1)
        scores, idxs = index.search(qv, args.k)
        print(f"\n=== {query}")
        for rank, (score, i) in enumerate(zip(scores[0], idxs[0]), 1):
            if i < 0:
                continue
            oa = "OA" if meta["is_open_access"][i] else "  "
            print(f"{rank:>2}. [{score:.3f}] {oa} {meta['title'][i]}")
            print(f"      {meta['journal'][i]} ({meta['year'][i]}) · "
                  f"{meta['cited_by_count'][i]} cites · {meta['doi'][i]}")


if __name__ == "__main__":
    main()
