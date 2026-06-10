"""
Shared helpers for the literature-index build pipeline.

Everything here mirrors the logic used to build the production CaBot index, so
that a third party can reproduce the *exact* embedding search engine locally.
Keep these helpers in sync with the docstrings in the numbered scripts.
"""

from configparser import ConfigParser
from pathlib import Path
import os

# --- Embedding configuration (must match the hosted index exactly) -----------
# The hosted index was built with OpenAI text-embedding-3-small at 1536 dims.
# Do NOT change these if you want bit-for-bit reproducibility of the paper index.
EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

# Documents longer than this many tokens are truncated head+tail before embedding
# (see build_document_text / truncate_tokens). 7000 was the production value.
TRUNCATE_TOKENS = 7000

# --- pgvector ANN configuration (must match the production server exactly) ----
# Verified against the live production database (PostgreSQL 17.5, pgvector 0.8.0):
#   CREATE INDEX vec_ann_idx ON works_vec
#       USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1732);
#   SET ivfflat.probes = 42;   -- set by the API before every search
# lists = sqrt(3M) ~= 1732; probes = sqrt(1732) ~= 42.
IVFFLAT_LISTS = 1732
IVFFLAT_PROBES = 42

# This module lives at <repo>/tools/build_literature_index/common.py.
# journals/ ships alongside it; build artifacts (data/) are written next to it
# and gitignored; secrets come from the repo-root config.ini.
PKG_DIR = Path(__file__).resolve().parent
REPO_ROOT = PKG_DIR.parent.parent
JOURNALS_DIR = PKG_DIR / "journals"
DATA_DIR = PKG_DIR / "data"


# --- Secrets -----------------------------------------------------------------
def _read_config():
    """Read the repo config.ini ([main] section). Looks next to these scripts
    first, then the repo root, so the same secrets file works either way."""
    cfg = ConfigParser()
    for candidate in (PKG_DIR / "config.ini", REPO_ROOT / "config.ini"):
        if candidate.exists():
            cfg.read(candidate)
            break
    return cfg


def _cfg_get(cfg, *names):
    """First non-empty value among [main] option names, else None."""
    if cfg.has_section("main"):
        for name in names:
            if cfg.has_option("main", name) and cfg.get("main", name).strip():
                return cfg.get("main", name).strip()
    return None


def get_embedding_client():
    """Return (OpenAI client, model_id) for generating embeddings.

    Reads the key from [main] OPENAI_API_KEY (or OPENAI_KEY) in config.ini, else
    the OPENAI_API_KEY environment variable.
    """
    import openai
    cfg = _read_config()
    oa_key = _cfg_get(cfg, "OPENAI_API_KEY", "OPENAI_KEY") or \
        os.environ.get("OPENAI_API_KEY", "").strip() or None
    if not oa_key:
        raise RuntimeError("No OpenAI key found (set [main] OPENAI_API_KEY in "
                           "config.ini or env OPENAI_API_KEY).")
    return openai.OpenAI(api_key=oa_key), EMBED_MODEL


def get_pg_dsn():
    """libpq DSN for the local pgvector database (04_load_postgres / 05_search).

    config.ini [main] PG_DSN, else env PG_DSN, else a local default. Any libpq
    connection string works, e.g. "dbname=cabot_search user=me password=... port=5432".
    """
    cfg = _read_config()
    return (_cfg_get(cfg, "PG_DSN")
            or os.environ.get("PG_DSN", "").strip()
            or "dbname=cabot_search host=localhost")


def get_hf_token():
    """HuggingFace write token: config.ini [main] HF_WRITE_TOKEN, else env HF_TOKEN."""
    cfg = _read_config()
    if cfg.has_section("main") and cfg.has_option("main", "HF_WRITE_TOKEN"):
        tok = cfg.get("main", "HF_WRITE_TOKEN").strip()
        if tok:
            return tok
    tok = os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        raise RuntimeError(
            "No HuggingFace token found. Set [main] HF_WRITE_TOKEN in config.ini "
            "or export HF_TOKEN."
        )
    return tok


# --- OpenAlex helpers --------------------------------------------------------
def reconstruct_abstract(inverted_index):
    """Rebuild plain-text abstract from OpenAlex's abstract_inverted_index.

    OpenAlex stores abstracts as {word: [positions]}; we invert that back into
    running text. Returns "" for missing/malformed indexes.
    """
    if not inverted_index:
        return ""
    try:
        size = max(p for ps in inverted_index.values() for p in ps) + 1
        words = [""] * size
        for word, positions in inverted_index.items():
            for p in positions:
                if 0 <= p < size:
                    words[p] = word
        # collapse the gaps left by any missing positions into single spaces
        return " ".join(" ".join(words).split())
    except (ValueError, TypeError):
        return ""


def truncate_tokens(text, tokenizer, max_tokens=TRUNCATE_TOKENS):
    """Head+tail truncate to max_tokens, matching the production ingest exactly."""
    encoded = tokenizer.encode(text)
    if len(encoded) <= max_tokens:
        return text
    half = max_tokens // 2
    head = tokenizer.decode(encoded[:half])
    tail = tokenizer.decode(encoded[-half:])
    return head + "...[truncated]..." + tail


def build_document_text(title, abstract, tokenizer):
    """Construct the text that gets embedded for a *document* (a work).

    Mirrors the production ingest:
      - title is lowercased + stripped
      - if an abstract exists: "<title>\\n\\n<abstract>", else title alone
      - NO "query: " prefix (that prefix is only applied to search queries)
      - head+tail truncation at TRUNCATE_TOKENS
    """
    title = (title or "").lower().strip()
    if abstract:
        text = f"{title}\n\n{abstract}"
    else:
        text = title
    return truncate_tokens(text, tokenizer)


def build_query_text(query):
    """Construct the text that gets embedded for a *search query*.

    Production prepends "query: " to queries (asymmetric with documents). To get
    comparable scores you MUST embed queries this way and documents without it.
    """
    return "query: " + query


# --- Parquet layout (single source of truth for the hosted dataset) ----------
# Column order/types of every shard. embedding is a fixed-size float32 vector.
# pyarrow is imported lazily so the lighter scripts (filter/search) don't pay for
# it unless they actually build/inspect the schema.
# biblio and oa_locations are stored as raw JSON strings (they back the
# nejmCitation / oaLocations / formattedOaLocations response fields).
PARQUET_COLUMNS = [
    "id", "doi", "title", "abstract", "journal", "year", "publication_date",
    "cited_by_count", "authors", "is_pubmed_indexed", "is_open_access",
    "article_type", "has_abstract", "biblio", "oa_locations", "embedding",
]


def parquet_schema():
    import pyarrow as pa
    return pa.schema([
        ("id", pa.string()),
        ("doi", pa.string()),
        ("title", pa.string()),
        ("abstract", pa.string()),
        ("journal", pa.string()),
        ("year", pa.int32()),
        ("publication_date", pa.string()),
        ("cited_by_count", pa.int32()),
        ("authors", pa.list_(pa.string())),
        ("is_pubmed_indexed", pa.bool_()),
        ("is_open_access", pa.bool_()),
        ("article_type", pa.string()),
        ("has_abstract", pa.bool_()),
        ("biblio", pa.string()),        # raw JSON, e.g. {"volume":...,"issue":...}
        ("oa_locations", pa.string()),  # raw JSON array, or null
        ("embedding", pa.list_(pa.float32(), EMBED_DIM)),
    ])
