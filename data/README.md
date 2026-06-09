# Data

The main-line CaBot versions (`v1`, `v1.1`) retrieve exemplar NEJM CPC differentials to ground their
style and reasoning. In this public release that comes from a single downloadable parquet:

```
data/
└── cpc_presentation_index_100.parquet   # 100 public CPCs: presentation embeddings + differentials
```

Download it from the **project website** (CPC-Bench dataset download page, "CaBot Exemplar Index" row)
and drop it here. Run with the default path, or point elsewhere with `--cpc-index`.

> **Scope note.** This index covers only the **100 public CPCs**. The original CaBot retrieved
> exemplars from the *full* (private) CPC corpus, so the cases retrieved here — and therefore the
> stylistic grounding — will differ from the runs reported in the paper. The retrieval mechanism is
> identical (same `text-embedding-3-small` embeddings, cosine search, year filtering); only the
> candidate pool is the 100-case public subset.

The parquet bundles, per case: `id`, `title`, `year`, `publication_date`, `presentation_of_case`,
`differential_diagnosis`, and the precomputed `embedding`. CaBot builds an in-memory ChromaDB
collection from it at startup and reads the differential text from the same file — no separate
ChromaDB directory or `cpcs_markdown_2.json` is needed.

Which versions need it:

| Version       | Needs `cpc_presentation_index_100.parquet`? |
|---------------|---------------------------------------------|
| `v1`, `v1.1`  | yes (exemplar retrieval)                    |
| `vr1`         | no — runs without local data                |
| `vs1`, `vs1.1`| no — simple QA / literature search only     |

(For `--mode video`/`both`, `--nejm-cpcs-path` should point at a directory of case images.)

The `literature_search` tool calls a hosted API and needs `JWT_CLINICTRON` in `config.ini`
for all versions (a fully local literature DB will be added later).
