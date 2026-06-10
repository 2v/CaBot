#!/usr/bin/env python3
"""
04_load_postgres.py -- load the index into local PostgreSQL + pgvector with the
EXACT production schema and ANN index configuration.

This recreates the database that the production CaBot /api/search endpoint
queries. Everything below was verified against the live production server
(PostgreSQL 17.5, pgvector 0.8.0) -- it matches `\\d works_meta_large` /
`\\d works_vec` there, column for column and index for index:

  * works_meta_large -- one row of display/filter metadata per work.
  * works_vec        -- one row per work: id + vector(1536) document embedding.
  * vec_ann_idx      -- the ANN index, exactly as production:
        CREATE INDEX vec_ann_idx ON works_vec
            USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1732);
  * the same seven btree indexes on works_meta_large (year, journal,
    has_abstract, cited_by_count, is_pubmed_indexed, is_open_access,
    article_type).

Known deltas from production (none affect search):
  * production's works_vec has a foreign key to a legacy `works_meta` table
    that is not part of this release; here it references works_meta_large.
  * `indexed_in` and `open_access` are loaded as NULL: they are not in the
    hosted dataset and the search API never reads them. The derived flags the
    API does return (`is_pubmed_indexed`, `is_open_access`) are in the dataset.

Embedding precision: the dataset ships float32 vectors and pgvector's `vector`
type stores float32, so the values loaded here are bit-identical to what
production stores (it inserted the same OpenAI outputs into vector(1536)).

REPRODUCIBILITY NOTE: IVFFlat index *builds* are not deterministic -- pgvector
trains the list centroids with k-means over a random sample. A rebuilt index is
therefore a different instance of the same method (lists=1732, probes=42): the
approximate top-5 can occasionally differ from the original server's specific
index, even though data, schema, and every parameter are identical.

Prereqs (Ubuntu; mirrors database/README.md in the parent repo):
    sudo apt install postgresql-17 postgresql-17-pgvector
    sudo -u postgres createdb cabot_search
    sudo -u postgres psql -d cabot_search -c "CREATE EXTENSION vector;"
    # plus access for your DSN user, e.g.:
    sudo -u postgres psql -d cabot_search -c "GRANT ALL ON SCHEMA public TO youruser;"
Set [main] PG_DSN in config.ini (or env PG_DSN) if the default
"dbname=cabot_search host=localhost" doesn't fit.

Usage:
    python 04_load_postgres.py                       # hosted index from HuggingFace
    python 04_load_postgres.py --local data/parquet  # shards you built yourself
    python 04_load_postgres.py --drop                # wipe + reload from scratch
    python 04_load_postgres.py --max-rows 100000     # quick partial load
    python 04_load_postgres.py --skip-index          # rows only; index separately

The full load is 3.47M rows. On the production machine (i9-9900K, 128 GB RAM)
the IVFFlat build alone took ~30 minutes with maintenance_work_mem=16GB and 8
parallel workers; tune --maintenance-work-mem / --parallel-workers to your
hardware (these affect build speed only, not the resulting search behavior).
"""

import argparse
import io
import sys
import time
from pathlib import Path

import psycopg2
import psycopg2.extras as extras
import pyarrow.dataset as ds

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import EMBED_DIM, IVFFLAT_LISTS, get_pg_dsn  # noqa: E402

DEFAULT_REPO = "tbuckley/cabot-search"

# Metadata columns read from the parquet shards (everything except `embedding`).
META_COLS = ["id", "year", "publication_date", "journal", "cited_by_count",
             "title", "abstract", "has_abstract", "is_pubmed_indexed",
             "is_open_access", "oa_locations", "authors", "doi", "biblio",
             "article_type"]

# DDL: identical to the production tables (see module docstring for the two
# documented deltas). Source of truth: live `\d` output + database/README.md.
DDL = """
CREATE TABLE IF NOT EXISTS works_meta_large (
    id                  text     PRIMARY KEY,   -- OpenAlex URI ("https://openalex.org/W...")
    year                smallint,
    publication_date    date,
    journal             text,
    cited_by_count      int,
    title               text,
    abstract            text,
    has_abstract        boolean NOT NULL,
    is_pubmed_indexed   boolean NOT NULL,
    is_open_access      boolean NOT NULL,
    indexed_in          text[],
    open_access         jsonb,
    oa_locations        jsonb,
    authors             text[],
    doi                 text,
    biblio              jsonb,
    article_type        text,
    created_at          timestamp with time zone DEFAULT now()
);

CREATE TABLE IF NOT EXISTS works_vec (
    id        text PRIMARY KEY REFERENCES works_meta_large(id) ON DELETE CASCADE,
    embedding vector(%d)
);
""" % EMBED_DIM

META_INDEXES = [
    "CREATE INDEX IF NOT EXISTS meta_year_idx    ON works_meta_large (year)",
    "CREATE INDEX IF NOT EXISTS meta_journal_idx ON works_meta_large (journal)",
    "CREATE INDEX IF NOT EXISTS meta_hasabs_idx  ON works_meta_large (has_abstract)",
    "CREATE INDEX IF NOT EXISTS meta_cited_idx   ON works_meta_large (cited_by_count)",
    "CREATE INDEX IF NOT EXISTS meta_pubmed_idx  ON works_meta_large (is_pubmed_indexed)",
    "CREATE INDEX IF NOT EXISTS meta_oa_idx      ON works_meta_large (is_open_access)",
    "CREATE INDEX IF NOT EXISTS meta_type_idx    ON works_meta_large (article_type)",
]

ANN_INDEX = ("CREATE INDEX vec_ann_idx ON works_vec "
             "USING ivfflat (embedding vector_cosine_ops) "
             "WITH (lists = %d)" % IVFFLAT_LISTS)


def log(msg):
    print(msg, flush=True)


def resolve_parquet_dir(args):
    """Return a local directory of parquet shards, downloading from HF if needed."""
    if args.local:
        d = Path(args.local)
        if not d.exists():
            sys.exit(f"Local dir not found: {d}")
        return d
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


def create_schema(conn, drop):
    cur = conn.cursor()
    try:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    except psycopg2.errors.InsufficientPrivilege:
        conn.rollback()
        cur.execute("SELECT 1 FROM pg_extension WHERE extname = 'vector'")
        if not cur.fetchone():
            sys.exit("pgvector is not installed in this database and this user "
                     "can't install it. Run as a superuser:\n"
                     "  psql -d <db> -c 'CREATE EXTENSION vector;'")
    if drop:
        log("Dropping existing tables (--drop)...")
        cur.execute("DROP TABLE IF EXISTS works_vec")
        cur.execute("DROP TABLE IF EXISTS works_meta_large CASCADE")
    cur.execute(DDL)
    conn.commit()


def batch_rows(batch):
    """One parquet record batch -> (meta_rows, vec_rows) ready for insertion."""
    cols = {c: batch.column(c).to_pylist() for c in META_COLS}
    flat = batch.column("embedding").flatten().to_numpy(zero_copy_only=False)
    mat = flat.reshape(-1, EMBED_DIM)
    meta_rows, vec_rows = [], []
    for i in range(len(batch)):
        meta_rows.append((
            cols["id"][i],
            cols["year"][i],
            cols["publication_date"][i] or None,   # 'YYYY-MM-DD' -> date
            cols["journal"][i],
            cols["cited_by_count"][i],
            cols["title"][i],
            cols["abstract"][i],
            cols["has_abstract"][i],
            cols["is_pubmed_indexed"][i],
            cols["is_open_access"][i],
            None,                                   # indexed_in (not in dataset)
            None,                                   # open_access (not in dataset)
            cols["oa_locations"][i],                # raw JSON string -> jsonb
            cols["authors"][i],
            cols["doi"][i],
            cols["biblio"][i],                      # raw JSON string -> jsonb
            cols["article_type"][i],
        ))
        # pgvector input format, same as the production ingest: '[v1,v2,...]'.
        # repr() round-trips each float32 exactly, so the stored vector is
        # bit-identical to the parquet value (and to production's).
        vec_rows.append((cols["id"][i],
                         "[" + ",".join(map(repr, mat[i].tolist())) + "]"))
    return meta_rows, vec_rows


def insert_batch(conn, meta_rows, vec_rows):
    """Insert one batch (meta via execute_values, vectors via COPY -- the same
    strategies the production ingest used). Idempotent: re-running skips rows
    that are already present."""
    cur = conn.cursor()
    extras.execute_values(
        cur,
        """
        INSERT INTO works_meta_large (
            id, year, publication_date, journal, cited_by_count,
            title, abstract, has_abstract, is_pubmed_indexed, is_open_access,
            indexed_in, open_access, oa_locations, authors, doi,
            biblio, article_type
        )
        VALUES %s
        ON CONFLICT DO NOTHING
        """,
        meta_rows, page_size=1000)

    buf = io.StringIO()
    for work_id, vec_str in vec_rows:
        buf.write(f"{work_id}\t{vec_str}\n")
    buf.seek(0)
    try:
        cur.copy_from(buf, "works_vec", columns=("id", "embedding"), sep="\t")
    except psycopg2.errors.UniqueViolation:
        # Re-run over existing rows: redo the whole batch with ON CONFLICT.
        conn.rollback()
        cur = conn.cursor()
        extras.execute_values(
            cur,
            """
            INSERT INTO works_meta_large (
                id, year, publication_date, journal, cited_by_count,
                title, abstract, has_abstract, is_pubmed_indexed, is_open_access,
                indexed_in, open_access, oa_locations, authors, doi,
                biblio, article_type
            )
            VALUES %s
            ON CONFLICT DO NOTHING
            """,
            meta_rows, page_size=1000)
        extras.execute_values(
            cur,
            "INSERT INTO works_vec (id, embedding) VALUES %s ON CONFLICT DO NOTHING",
            vec_rows, template="(%s, %s::vector)", page_size=1000)
    conn.commit()


def build_indexes(conn, args):
    cur = conn.cursor()
    log("Creating metadata btree indexes...")
    for q in META_INDEXES:
        cur.execute(q)
    conn.commit()

    log(f"Building IVFFlat index (lists = {IVFFLAT_LISTS}); this can take a while...")
    # Session settings for the build (production used 16GB / 8 workers; these
    # only affect build speed, not search behavior).
    cur.execute(f"SET maintenance_work_mem = '{args.maintenance_work_mem}'")
    cur.execute(f"SET max_parallel_maintenance_workers = {args.parallel_workers}")
    cur.execute("DROP INDEX IF EXISTS vec_ann_idx")
    t0 = time.time()
    cur.execute(ANN_INDEX)
    conn.commit()
    log(f"IVFFlat index built in {time.time() - t0:.0f}s.")

    log("Running ANALYZE...")
    cur.execute("ANALYZE works_meta_large")
    cur.execute("ANALYZE works_vec")
    conn.commit()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--local", help="Local parquet dir (from 02_build_embeddings.py).")
    src.add_argument("--repo-id", help=f"HuggingFace dataset repo (default: {DEFAULT_REPO}).")
    ap.add_argument("--dsn", help="libpq DSN (default: config.ini [main] PG_DSN / env PG_DSN).")
    ap.add_argument("--drop", action="store_true",
                    help="Drop + recreate the tables before loading.")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Load only this many rows.")
    ap.add_argument("--batch-size", type=int, default=10_000,
                    help="Rows per read/insert batch. Bounds peak memory: each batch is "
                         "materialized as Python objects + a text COPY buffer, so large "
                         "batches (parquet row-group sized) can OOM small machines.")
    ap.add_argument("--skip-index", action="store_true",
                    help="Load rows but skip index creation.")
    ap.add_argument("--maintenance-work-mem", default="2GB",
                    help="maintenance_work_mem for the index build (default: 2GB; "
                         "production used 16GB).")
    ap.add_argument("--parallel-workers", type=int, default=4,
                    help="max_parallel_maintenance_workers for the index build "
                         "(default: 4; production used 8).")
    args = ap.parse_args()
    if not args.local and not args.repo_id:
        args.repo_id = DEFAULT_REPO

    parquet_dir = resolve_parquet_dir(args)
    dsn = args.dsn or get_pg_dsn()
    log(f"Connecting to: {dsn}")
    conn = psycopg2.connect(dsn)

    create_schema(conn, args.drop)

    dataset = ds.dataset(str(parquet_dir), format="parquet")
    log(f"Loading rows from {parquet_dir} ...")
    loaded = 0
    t0 = time.time()
    for batch in dataset.to_batches(columns=META_COLS + ["embedding"],
                                    batch_size=args.batch_size):
        if args.max_rows and loaded + len(batch) > args.max_rows:
            batch = batch.slice(0, args.max_rows - loaded)
        meta_rows, vec_rows = batch_rows(batch)
        insert_batch(conn, meta_rows, vec_rows)
        loaded += len(meta_rows)
        rate = loaded / (time.time() - t0)
        log(f"  {loaded:,} rows loaded ({rate:,.0f} rows/s)")
        if args.max_rows and loaded >= args.max_rows:
            break
    log(f"Done loading {loaded:,} rows in {time.time() - t0:.0f}s.")

    if args.skip_index:
        log("--skip-index: skipping index creation.")
    else:
        build_indexes(conn, args)

    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM works_vec")
    log(f"works_vec now holds {cur.fetchone()[0]:,} vectors. "
        f"Search with: python 05_search.py --query '...'")
    conn.close()


if __name__ == "__main__":
    main()
