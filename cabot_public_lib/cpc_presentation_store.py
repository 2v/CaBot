"""
Presentation-of-case exemplar store for CaBot-Public.

This store is built from a single parquet of precomputed presentation-of-case embeddings; the
vectors live in memory as a numpy matrix — no vector-database dependency. It loads whichever
parquet it is given, with the same schema either way:

  * ``cpc_presentation_index_100.parquet`` — the 100 public CPCs shipped with this release, OR
  * ``cpc_presentation_index_full.parquet`` — the private full >6,000-case corpus CaBot v1
    actually retrieved over (pulled via ``fetch_data.py --full-cpc-index`` when authenticated).

NOTE: with the public 100-case index, retrieved exemplars differ from the paper's runs because the
original CaBot retrieved over the full (private) CPC corpus; loading the full index reproduces that.

The query path embeds the search text with ``text-embedding-3-small`` — the same model used to build
the index — so the stored and query vectors are comparable. Similarity is cosine: vectors are
L2-normalized and compared by inner product, so ``score`` is the cosine similarity (identical to the
``1 - cosine_distance`` the previous ChromaDB-backed store returned).

Like ``LiteratureSearchStore``, the store loads its data in its constructor: pass a parquet path
(or pre-loaded records) and it comes back ready to search, also exposing ``records``,
``titles_by_id`` and ``ddx_by_id`` for the caller.
"""
import sys

import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq

from .openai_retry import call_with_retry

EMBED_MODEL = "text-embedding-3-small"


def load_index_parquet(parquet_path):
    """Load the public CPC exemplar index parquet into a list of record dicts."""
    return pq.read_table(parquet_path).to_pylist()


def _normalize_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > 0)


class CPCPresentationStore:
    def __init__(self, openai_client, cpc_index_path=None, records=None, verbose=True):
        self.openai_client = openai_client
        self.verbose = verbose
        # A normalized float32 matrix + parallel metadata.
        self.matrix = None          # (N, dim) L2-normalized float32
        self.ids = []
        self.documents = []
        self.metadatas = []
        self.records = []           # raw parquet records (titles / DDx / ids)
        self.titles_by_id = {}
        self.ddx_by_id = {}
        self._years = None          # int array, None -> sentinel
        self._has_year = None

        # Self-load in the constructor (parity with LiteratureSearchStore): given a
        # parquet path or pre-loaded records, the store comes back ready to search.
        if records is None and cpc_index_path is not None:
            records = load_index_parquet(cpc_index_path)
        if records is not None:
            self.build_from_records(records)

    def get_embeddings(self, texts, batch_size=30):
        """Embed query text with the same model used to build the index."""
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", disable=len(texts) <= batch_size):
            batch = texts[i:i + batch_size]
            response = call_with_retry(self.openai_client.embeddings.create,
                                       input=batch, model=EMBED_MODEL)
            all_embeddings.extend(d.embedding for d in response.data)
        return all_embeddings

    def build_from_records(self, records):
        """Build the in-memory matrix from parquet records (precomputed embeddings, no re-embed)."""
        embeddings, years = [], []
        self.ids, self.documents, self.metadatas = [], [], []
        self.titles_by_id, self.ddx_by_id = {}, {}
        for r in records:
            case_id = r["id"]
            year = r.get("year")
            decade = r.get("decade")
            self.ids.append(case_id)
            self.documents.append(r.get("presentation_of_case", ""))
            embeddings.append(list(r["embedding"]))
            years.append(int(year) if year is not None else None)
            self.metadatas.append({
                "year": int(year) if year is not None else "unknown",
                "decade": int(decade) if decade is not None else "unknown",
                "title": r.get("title", "") or "",
                "publication_date": r.get("publication_date", "") or "",
                "case_id": case_id,
            })
            self.titles_by_id[case_id] = r.get("title") or f"Case {case_id}"
            self.ddx_by_id[case_id] = r.get("differential_diagnosis")

        mat = np.asarray(embeddings, dtype=np.float32)
        self.matrix = _normalize_rows(mat)
        self._has_year = np.array([y is not None for y in years], dtype=bool)
        self._years = np.array([y if y is not None else 0 for y in years], dtype=np.int64)
        self.records = records
        if self.verbose:
            print(f"Loaded {len(records)} CPC exemplars.", file=sys.stderr, flush=True)
        return self

    def build_from_parquet(self, parquet_path):
        """Convenience: load the parquet and build the collection."""
        return self.build_from_records(load_index_parquet(parquet_path))

    def search_with_filters(self, query, k=5, year_min=None, year_max=None, decade=None, exclude_id=None):
        """Embed the query and search with optional year/decade/exclude filters.

        Returns (scores, case_ids, documents, metadatas), scores being cosine similarities.
        Filter semantics match the previous ChromaDB store: a row with no year is excluded by
        any year filter (it was stored as "unknown").
        """
        if self.matrix is None:
            raise ValueError("Index not built. Call build_from_parquet()/build_from_records() first.")

        try:
            q = np.asarray(self.get_embeddings([query])[0], dtype=np.float32)
            qnorm = float(np.linalg.norm(q))
            if qnorm > 0:
                q = q / qnorm

            mask = np.ones(len(self.ids), dtype=bool)
            if year_min is not None:
                mask &= self._has_year & (self._years >= int(year_min))
            if year_max is not None:
                mask &= self._has_year & (self._years <= int(year_max))
            if decade is not None:
                decades = np.array(
                    [m["decade"] if isinstance(m["decade"], int) else None for m in self.metadatas])
                mask &= np.array([d == decade for d in decades], dtype=bool)
            if exclude_id is not None:
                mask &= np.array([cid != exclude_id for cid in self.ids], dtype=bool)

            eligible = np.nonzero(mask)[0]
            if eligible.size == 0:
                return [], [], [], []

            scores_all = self.matrix[eligible] @ q
            kk = min(k, eligible.size)
            top = np.argpartition(-scores_all, kk - 1)[:kk]
            top = top[np.argsort(-scores_all[top])]
            sel = eligible[top]

            scores = [float(s) for s in scores_all[top]]
            case_ids = [self.metadatas[i]["case_id"] for i in sel]
            documents = [self.documents[i] for i in sel]
            metadatas = [self.metadatas[i] for i in sel]
            return scores, case_ids, documents, metadatas
        except Exception as e:
            print(f"Search failed: {e}")
            return [], [], [], []
