# CaBot-Search

A self-contained reproduction of the clinical-literature embedding-search engine
used for CaBot.

It is an **exact embedding search index** over **3,474,244 works** from **204
high-impact clinical journals** (2023 Journal Impact Factor ≥ 10), embedded with
OpenAI `text-embedding-3-small` (1536-dim, float32). This folder contains
everything needed to either **(a) search the hosted index locally** or **(b)
rebuild the entire index from a raw OpenAlex snapshot.**

> **Provenance.** The hosted index was first built in **early June 2025**
> (OpenAlex snapshot ~2025-06-05). OpenAlex changes over time, so rebuilding from
> a newer snapshot yields a similar but not byte-identical subset. For exact
> paper reproducibility, use an OpenAlex snapshot from on/around 2025-06-05.

```
CaBot-Search/
├── README.md
├── requirements.txt
├── config.ini.example          # OpenAI + HuggingFace keys
├── journals/                    # the journal selection (provenance)
│   ├── high_impact_journals_if10.txt   # 204 journals, 2023 JIF ≥ 10
│   ├── top_journals_jcr_2023.txt       # raw JCR export with IF values
│   └── journal_source_ids.json         # journal → OpenAlex source id mapping
└── scripts/
    ├── common.py                # shared config + embedding helpers
    ├── 01_filter_openalex.py    # raw OpenAlex snapshot → journal subset
    ├── 02_build_embeddings.py   # subset → embedded parquet shards (float32)
    ├── 03_upload_huggingface.py # publish parquet to a HuggingFace dataset
    └── 04_search.py             # load index → exact FAISS search → sample queries
```

---

## Setup

```bash
cd CaBot-Search
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp config.ini.example config.ini   # then add your keys
```

`config.ini` keys (or use the matching env vars):

| key (`[main]`) | env var | used by |
|---|---|---|
| `OPENROUTER_KEY` *or* `OPENAI_KEY` | `OPENROUTER_API_KEY` / `OPENAI_API_KEY` | `02_build_embeddings.py`, `04_search.py` (query embeddings) |
| `HF_WRITE_TOKEN` | `HF_TOKEN` | `03_upload_huggingface.py` |

Embeddings can be generated via **OpenRouter** (an OpenAI-compatible gateway) or
**OpenAI** directly — both route to the same `text-embedding-3-small` and yield
identical 1536-d vectors. Provide whichever key you have; set `EMBED_PROVIDER`
(`openrouter`|`openai`) to force one when both are present.

---

## 1. Quickstart — search the hosted index locally

The full index is published at
**[huggingface.co/datasets/tbuckley/cabot-search](https://huggingface.co/datasets/tbuckley/cabot-search)**
(public, ~21 GB across 14 parquet shards). `04_search.py` downloads it (cached
after the first run), builds an **exact** FAISS cosine index, and runs a few
sample queries:

```bash
python scripts/04_search.py                       # defaults to tbuckley/cabot-search
# equivalently: python scripts/04_search.py --repo-id tbuckley/cabot-search
```

Example output (full 3.47M-doc index):

```
=== GLP-1 receptor agonists and cardiovascular outcomes in type 2 diabetes
 1. [0.831]    GLP-1 RECEPTOR AGONISTS AND CARDIOVASCULAR OUTCOMES IN PATIENTS WITH ATRIAL FIBRILLATION AND DIABETES
       Journal of the American College of Cardiology (2024) · 0 cites · https://doi.org/10.1016/s0735-1097(24)02182-x
 2. [0.817] OA Cardiovascular, mortality, and kidney outcomes with GLP-1 receptor agonists in patients with type 2 diabetes: ...
       The Lancet Diabetes & Endocrinology (2019) · 1299 cites · https://doi.org/10.1016/s2213-8587(19)30249-9
 ...
```

Try your own query / more results:

```bash
python scripts/04_search.py --query "CAR-T therapy for relapsed B-cell lymphoma" --k 10
```

> **Hardware.** The full index is 3.47M × 1536 float32 ≈ **21 GB in RAM** (plus an
> equal-size copy inside the FAISS index — budget ~45 GB). On a smaller machine,
> load a subset with `--max-rows 200000` (results are then drawn from that subset
> only), or point `--local` at parquet shards you built yourself.

### How search works (matches production)

- **Documents** are embedded from `title.lower()`, or `title.lower() + "\n\n" +
  abstract` when an abstract exists (head+tail truncated to 7000 tokens). **No**
  `query:` prefix.
- **Queries** are embedded as `"query: " + query`.
- **Similarity is cosine.** Vectors are L2-normalized and compared with inner
  product (`faiss.IndexFlatIP`), so the score equals cosine similarity. Production
  reports `1 − cosine_distance`, the same quantity, but uses an approximate
  IVFFlat index for speed; this exact index gives identical-or-better recall.

### `--json`: the exact CaBot `/api/search` response

`--json` emits the **same JSON response as the CaBot `/api/search` API**, with the
same fields, NEJM author/citation formatting, OA-location formatting, dual
abstract/title retrieval, and the same filters:

```bash
python scripts/04_search.py --query "statin primary prevention" --json \
    --need-abstract --year-from 2015 --year-to 2023 \
    --citations-min 50 --journals "The Lancet,New England Journal of Medicine"
```

Mapping to the API query parameters: `--need-abstract`→`needAbstract=true`,
`--year-from/--year-to`→`yearFrom/yearTo`, `--citations-min/--citations-max`→
`citationsMin/citationsMax`, `--journals`→`journals` (comma-separated). As in the
API, `needAbstract=false` returns up to 10 results (5 without an abstract, then 5
with); `needAbstract=true` returns 5.

The output is **byte-for-byte identical to the API** — verified against the
production controller — with two unavoidable exceptions from using an exact local
backend instead of the API's approximate pgvector index:

1. **`score`** agrees to ~1e-7 but not in its last digits (numpy vs. pgvector
   floating-point arithmetic).
2. **Result membership** can differ slightly: the API's *approximate* IVFFlat
   index occasionally misses a document that this *exact* search returns (local is
   the more-correct result). Every overlapping document is byte-identical.

### Dataset schema

| column | type | notes |
|---|---|---|
| `id` | string | OpenAlex work id |
| `doi` | string | |
| `title` | string | |
| `abstract` | string | reconstructed from OpenAlex inverted index (`""` if none) |
| `journal` | string | |
| `year` | int32 | |
| `publication_date` | string | `YYYY-MM-DD` |
| `cited_by_count` | int32 | |
| `authors` | list[string] | raw author names |
| `is_pubmed_indexed` | bool | |
| `is_open_access` | bool | |
| `article_type` | string | OpenAlex `type` |
| `has_abstract` | bool | |
| `biblio` | string (JSON) | `{"volume","issue","first_page","last_page"}`; backs `nejmCitation` |
| `oa_locations` | string (JSON) | open-access locations array, or null; backs `oaLocations`/`formattedOaLocations` |
| `embedding` | list[float32]×1536 | `text-embedding-3-small` document embedding |

Load it directly if you want to build your own index (FAISS, hnswlib, etc.):

```python
import pyarrow.dataset as ds
table = ds.dataset("data/parquet", format="parquet")   # shards = one logical table
# or: from datasets import load_dataset; load_dataset("<repo-id>", split="train")
```

---

## 2. Building the index from scratch

This reproduces the full pipeline: raw OpenAlex → journal subset → embeddings →
published dataset. Budget for a large download, ~21 GB of vectors, and a few
billion OpenAI embedding tokens (check current pricing before a full run).

### Prerequisites: download an OpenAlex snapshot

OpenAlex publishes a free, no-auth S3 snapshot. We only need the `works` tree:

```bash
aws s3 sync "s3://openalex/data/works" "openalex-snapshot/data/works" --no-sign-request
```

For exact reproducibility, use a snapshot from ~2025-06-05 (see Provenance above).

### How the journal subset is defined

The journal universe is the **204 titles with 2023 JIF ≥ 10** in
`journals/high_impact_journals_if10.txt`, derived from the Clarivate JCR export in
`journals/top_journals_jcr_2023.txt`. Each title is matched to its OpenAlex
**source id(s)** in `journals/journal_source_ids.json` (the original mapping was
built by scanning the OpenAlex `sources/` tree; the result is checked in, so you
don't need `sources/` to rebuild). A work is kept if **any** of its locations —
primary or secondary — points at one of those source ids.

### Step 1 — filter the snapshot to the journal subset

```bash
python scripts/01_filter_openalex.py \
    --snapshot openalex-snapshot/data/works
# → data/journal_works.jsonl.gz   (~3.5M works, abstracts pre-reconstructed)
```

Add `--limit 100000` for a fast smoke test.

### Step 2 — embed and export to parquet

```bash
python scripts/02_build_embeddings.py        # → data/parquet/part-00000.parquet, ...
```

Each work is embedded with `text-embedding-3-small` (1536-d, float32) and written
as **sharded** Parquet (default 250k rows/shard) so the ~25–30 GB output uploads,
downloads, and resumes reliably. Useful flags:

- `--limit 5000` — quick test
- `--resume` — continue an interrupted run (skips works in completed shards)
- `--shard-size N` / `--embed-batch N` — tune shard size / API batch size

### Step 3 — publish to HuggingFace

```bash
python scripts/03_upload_huggingface.py --repo-id <your-username>/cabot-search
```

Creates the dataset repo (if needed), writes a dataset card, and resumably
uploads the parquet shards plus the `journals/` provenance files over Git-LFS. Add
`--private` to keep it private. Point your website's download link (and
`04_search.py --repo-id`) at the resulting repo.

---

## Notes & caveats

- **Exact vs. approximate.** This standalone uses brute-force exact cosine search.
  The production site uses an approximate IVFFlat index purely for latency; the
  document embeddings are identical.
- **Float32.** Embeddings are stored at full precision for bit-exact
  reproducibility. float16 would roughly halve the size with negligible recall
  loss if download size matters more than exactness.
- **Query/document asymmetry.** The `"query: "` prefix is applied to queries only.
  Embedding documents with the prefix (or queries without it) will shift scores.
- **No abstract?** Works without an abstract are still embedded (title only) and
  included, with `has_abstract = false`.
