#!/usr/bin/env python3
"""
02_build_embeddings.py  --  Journal subset -> embedded, hostable dataframe.

SECOND stage. Reads the filtered works from stage 1 (journal_works.jsonl.gz),
embeds each one with OpenAI text-embedding-3-small (1536-dim, float32), and
writes everything a user needs to host their own exact embedding-search index:
doi, title, abstract, journal, year, citation count, authors, ... AND the raw
1536-d embedding, as sharded Parquet.

------------------------------------------------------------------------------
WHY SHARDED PARQUET (and why it's still "one file" to load)
------------------------------------------------------------------------------
~3.5M works * 1536 float32 = ~21 GB of vectors (plus abstracts) -> ~25-30 GB.
A single 30 GB Parquet is painful to upload/download/resume, so we write a
directory of equal-sized shards: data/parquet/part-00000.parquet, part-00001...
HuggingFace, pandas (via pyarrow.dataset), and `datasets.load_dataset` all treat
a directory of parquet shards as ONE logical table, so loading is still one line.
Embeddings are kept at full float32 precision for bit-exact reproducibility.

------------------------------------------------------------------------------
EMBEDDING DETAILS (must match the hosted index)
------------------------------------------------------------------------------
  model         text-embedding-3-small
  dimensions    1536
  document text title.lower() , or title.lower() + "\\n\\n" + abstract
                (NO "query: " prefix -- that is only for search queries)
  truncation    head+tail to 7000 tokens
See common.py (alongside this script) for the exact helpers.

Usage:
    python 02_build_embeddings.py                       # full run
    python 02_build_embeddings.py --limit 5000          # quick test
    python 02_build_embeddings.py --resume              # continue an interrupted run
    python 02_build_embeddings.py --shard-size 250000   # rows per parquet shard

Cost note: embedding ~3.5M works with text-embedding-3-small is on the order of
a few billion tokens. Check current OpenAI pricing before a full run.
"""

import argparse
import gzip
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import openai
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    DATA_DIR, EMBED_DIM, EMBED_MODEL, build_document_text, get_embedding_client,
    parquet_schema, reconstruct_abstract,
)

DEFAULT_INPUT = DATA_DIR / "journal_works.jsonl.gz"
DEFAULT_OUTDIR = DATA_DIR / "parquet"

# OpenAI embeddings API request limits. We pack requests up to both bounds.
MAX_INPUTS_PER_REQUEST = 2000
MAX_TOKENS_PER_REQUEST = 230_000
THREADS = 16

tokenizer = tiktoken.encoding_for_model(EMBED_MODEL)
SCHEMA = parquet_schema()

# Embedding client + model id are resolved in main().
CLIENT = None
MODEL_ID = None


def count_tokens(text):
    return len(tokenizer.encode(text))


def extract_metadata(work):
    """Pull the display/metadata fields we host alongside each embedding."""
    pub_date = work.get("publication_date")  # already 'YYYY-MM-DD' or None
    authors = [
        a.get("raw_author_name")
        for a in (work.get("authorships") or [])
        if a.get("raw_author_name")
    ]
    indexed_in = work.get("indexed_in") or []
    journal = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
    abstract = work.get("abstract")
    if abstract is None:  # stage 1 normally fills this; recompute if absent
        abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
    # biblio + the open-access subset of locations back the citation/OA fields.
    biblio = work.get("biblio")
    oa_locs = [loc for loc in (work.get("locations") or []) if loc.get("is_oa")] or None
    return {
        "id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("display_name"),
        "abstract": abstract,
        "journal": journal,
        "year": work.get("publication_year"),
        "publication_date": pub_date,
        "cited_by_count": work.get("cited_by_count"),
        "authors": authors,
        "is_pubmed_indexed": "pubmed" in indexed_in,
        "is_open_access": bool((work.get("open_access") or {}).get("is_oa")),
        "article_type": work.get("type") or "unknown",
        "has_abstract": bool(abstract),
        "biblio": orjson.dumps(biblio).decode() if biblio else None,
        "oa_locations": orjson.dumps(oa_locs).decode() if oa_locs else None,
    }


def embed_texts(texts):
    """Embed a list of texts, packing them into token/-input-bounded API calls
    run in parallel. Returns embeddings aligned 1:1 with `texts`."""
    # Build chunks (index lists) that respect both API limits.
    chunks, cur, cur_tokens = [], [], 0
    for i, text in enumerate(texts):
        t = min(count_tokens(text), MAX_TOKENS_PER_REQUEST)
        if cur and (len(cur) >= MAX_INPUTS_PER_REQUEST or cur_tokens + t > MAX_TOKENS_PER_REQUEST):
            chunks.append(cur)
            cur, cur_tokens = [], 0
        cur.append(i)
        cur_tokens += t
    if cur:
        chunks.append(cur)

    out = [None] * len(texts)

    def run_chunk(indices):
        """Embed one chunk; split-and-retry on the token-limit error."""
        sub = [texts[i] for i in indices]
        try:
            resp = CLIENT.embeddings.create(model=MODEL_ID, input=sub, dimensions=EMBED_DIM)
            for j, item in enumerate(resp.data):
                out[indices[j]] = item.embedding
        except openai.BadRequestError as e:
            if "max_tokens_per_request" in str(e) and len(indices) > 1:
                mid = len(indices) // 2
                run_chunk(indices[:mid])
                run_chunk(indices[mid:])
            else:
                raise

    with ThreadPoolExecutor(max_workers=min(THREADS, len(chunks))) as ex:
        list(ex.map(run_chunk, chunks))
    return out


def write_shard(outdir, shard_idx, rows):
    """Write one buffered batch of rows to part-NNNNN.parquet."""
    table = pa.Table.from_pylist(rows, schema=SCHEMA)
    path = outdir / f"part-{shard_idx:05d}.parquet"
    pq.write_table(table, path, compression="zstd")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT,
                    help=f"Filtered works from stage 1 (default: {DEFAULT_INPUT})")
    ap.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR,
                    help=f"Output parquet directory (default: {DEFAULT_OUTDIR})")
    ap.add_argument("--shard-size", type=int, default=250_000,
                    help="Rows per parquet shard (default: 250000).")
    ap.add_argument("--embed-batch", type=int, default=2000,
                    help="Works embedded per OpenAI round-trip batch (default: 2000).")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after embedding this many works (for quick tests).")
    ap.add_argument("--resume", action="store_true",
                    help="Skip works already covered by existing complete shards.")
    args = ap.parse_args()

    if not args.input.exists():
        ap.error(f"Input not found: {args.input}. Run 01_filter_openalex.py first.")
    args.outdir.mkdir(parents=True, exist_ok=True)

    global CLIENT, MODEL_ID
    CLIENT, MODEL_ID = get_embedding_client()
    print(f"Embedding via OpenAI model id '{MODEL_ID}'.")

    # Resume: each completed shard holds exactly --shard-size works (except the
    # last). Count complete shards and skip that many input lines.
    existing = sorted(args.outdir.glob("part-*.parquet"))
    shard_idx = 0
    skip = 0
    if args.resume and existing:
        # Treat all but the last shard as full; drop a possibly-partial last shard
        # by recounting. Simpler + safe: keep full-size shards only.
        full = [p for p in existing if pq.ParquetFile(p).metadata.num_rows == args.shard_size]
        shard_idx = len(full)
        skip = shard_idx * args.shard_size
        for p in existing[len(full):]:
            p.unlink()  # remove trailing partial shard; it will be rebuilt
        print(f"Resume: {shard_idx} full shards found, skipping first {skip:,} works.")

    buffer, embed_texts_buf, embed_meta_buf = [], [], []
    total = 0
    start = time.time()

    def flush_embeddings():
        """Embed the pending batch and append finished rows to `buffer`."""
        if not embed_texts_buf:
            return
        vectors = embed_texts(embed_texts_buf)
        for meta, vec in zip(embed_meta_buf, vectors):
            if vec is None:
                continue
            buffer.append({**meta, "embedding": vec})
        embed_texts_buf.clear()
        embed_meta_buf.clear()

    print(f"Embedding works from {args.input} ...")
    with gzip.open(args.input, "rb") as f:
        for line_no, line in enumerate(tqdm(f, desc="works", unit=" works")):
            if line_no < skip:
                continue
            work = orjson.loads(line)
            meta = extract_metadata(work)
            if not meta["title"]:
                continue  # production skips untitled works
            text = build_document_text(meta["title"], meta["abstract"], tokenizer)
            if not text.strip():
                continue
            embed_meta_buf.append(meta)
            embed_texts_buf.append(text)

            if len(embed_texts_buf) >= args.embed_batch:
                flush_embeddings()

            while len(buffer) >= args.shard_size:
                path = write_shard(args.outdir, shard_idx, buffer[:args.shard_size])
                del buffer[:args.shard_size]
                print(f"  wrote {path.name}")
                shard_idx += 1

            total += 1
            if args.limit and total >= args.limit:
                break

    flush_embeddings()
    if buffer:
        path = write_shard(args.outdir, shard_idx, buffer)
        print(f"  wrote {path.name} (final, {len(buffer):,} rows)")

    elapsed = time.time() - start
    print(f"\nDone. Embedded {total:,} works in {elapsed/60:.1f} min "
          f"-> {args.outdir} ({EMBED_MODEL}, {EMBED_DIM}-d float32)")
    print(f"Build timestamp: {datetime.now().isoformat(timespec='seconds')}")
    print("Next: 03_upload_huggingface.py to publish, or 04_load_postgres.py + "
          "05_search.py to load + search locally.")


if __name__ == "__main__":
    main()
