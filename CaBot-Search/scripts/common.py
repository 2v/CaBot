"""
Shared helpers for the CaBot-Search standalone pipeline.

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

# Embeddings can be generated through OpenAI directly, or through OpenRouter
# (an OpenAI-compatible gateway). OpenRouter routes to the same underlying model,
# so the vectors are identical to OpenAI's (verified: cosine 1.0000 vs the hosted
# index). See get_embedding_client() for provider selection.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "openai/text-embedding-3-small"  # OpenRouter's id for the same model

# Documents longer than this many tokens are truncated head+tail before embedding
# (see build_document_text / truncate_tokens). 7000 was the production value.
TRUNCATE_TOKENS = 7000

# Repo root is one level up from this scripts/ directory.
ROOT = Path(__file__).resolve().parent.parent
JOURNALS_DIR = ROOT / "journals"
DATA_DIR = ROOT / "data"


# --- Secrets -----------------------------------------------------------------
def _read_config():
    """Read the repo config.ini ([main] section). Looks in CaBot-Search/ first,
    then the parent OOE repo root, so the same secrets file works everywhere."""
    cfg = ConfigParser()
    for candidate in (ROOT / "config.ini", ROOT.parent / "config.ini"):
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
    """Return (OpenAI-compatible client, model_id) for generating embeddings.

    Works with either provider:
      - OpenRouter: [main] OPENROUTER_KEY, or env OPENROUTER_API_KEY
        -> model "openai/text-embedding-3-small"
      - OpenAI:     [main] OPENAI_KEY, or env OPENAI_API_KEY
        -> model "text-embedding-3-small"

    Provider is chosen by [main] EMBED_PROVIDER / env EMBED_PROVIDER
    ("openrouter" | "openai") if set; otherwise it auto-selects OpenRouter when an
    OpenRouter key is present, else OpenAI. Both yield identical 1536-d vectors.
    """
    import openai
    cfg = _read_config()
    or_key = _cfg_get(cfg, "OPENROUTER_KEY") or \
        os.environ.get("OPENROUTER_API_KEY", "").strip() or None
    oa_key = _cfg_get(cfg, "OPENAI_KEY") or \
        os.environ.get("OPENAI_API_KEY", "").strip() or None

    provider = (_cfg_get(cfg, "EMBED_PROVIDER")
                or os.environ.get("EMBED_PROVIDER", "").strip().lower() or None)
    if provider is None:
        provider = "openrouter" if or_key else "openai"

    if provider == "openrouter":
        if not or_key:
            raise RuntimeError("EMBED_PROVIDER=openrouter but no OpenRouter key found "
                               "(set [main] OPENROUTER_KEY or env OPENROUTER_API_KEY).")
        return openai.OpenAI(api_key=or_key, base_url=OPENROUTER_BASE_URL), OPENROUTER_MODEL
    if provider == "openai":
        if not oa_key:
            raise RuntimeError("EMBED_PROVIDER=openai but no OpenAI key found "
                               "(set [main] OPENAI_KEY or env OPENAI_API_KEY).")
        return openai.OpenAI(api_key=oa_key), EMBED_MODEL
    raise RuntimeError(f"Unknown EMBED_PROVIDER={provider!r} (use 'openrouter' or 'openai').")


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
