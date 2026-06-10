"""
Literature-search store for CaBot-Public — the exact production pgvector engine.

This is the self-hosted backend for CaBot's ``literature_search`` tool. It queries a local
**PostgreSQL + pgvector** database that holds the clinical-literature index (3.47M works from
204 high-impact journals, embedded with ``text-embedding-3-small`` at 1536 dims). The search
runs the **same SQL as the production CaBot /api/search endpoint**, against the same schema
(``works_meta_large`` + ``works_vec``), over the same approximate ANN index:

    CREATE INDEX vec_ann_idx ON works_vec
        USING ivfflat (embedding vector_cosine_ops) WITH (lists = 1732);
    SET ivfflat.probes = 42;            -- set on the session before searching
    score = 1 - (embedding <=> query)   -- cosine similarity

so results reproduce the live site's behavior rather than approximating it with a different
backend. Load the database once with ``tools/build_literature_index/04_load_postgres.py``
(downloads the index from HuggingFace and builds the identical IVFFlat index); the vectors live
on disk in Postgres, so the store needs no large in-RAM index.

Query/document asymmetry: documents were embedded WITHOUT a prefix; queries are embedded as
``"query: " + text``. The IVFFlat index is approximate and its build is randomized, so a freshly
loaded index is a different instance of the same method — the top-5 can occasionally differ from
the original server's specific index, but the method and every parameter are identical.
"""
import json
import sys

import psycopg2
import psycopg2.extras

from .openai_retry import call_with_retry

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536
DEFAULT_PG_DSN = "dbname=cabot_search host=localhost"
IVFFLAT_PROBES = 42

# The SELECT list, join, and ordering are a literal transcription of the production
# controller (ooe_site/server/src/controllers/search.js), identical to
# tools/build_literature_index/05_search.py.
BASE_SELECT = """
        SELECT m.id,
               m.title,
               1 - (v.embedding <=> %(qvec)s) AS score,
               m.year,
               m.journal,
               m.has_abstract,
               m.abstract,
               m.oa_locations,
               m.cited_by_count,
               m.authors,
               m.doi,
               m.biblio,
               m.publication_date,
               m.is_pubmed_indexed,
               m.is_open_access,
               m.article_type
        FROM   works_meta_large AS m
        JOIN   works_vec  AS v USING (id)
        WHERE  TRUE
"""


def format_authors_nejm(authors):
    """'Last F.M.' style, first 6 authors, '..., et al' if >6.

    Ported verbatim from the index-build tooling (apiformat.format_authors_nejm) so the
    author strings match the rest of the pipeline, including its JavaScript-quirk behavior
    on empty name parts.
    """
    if not authors:
        return "Unknown authors"
    try:
        formatted = []
        for name in authors[:6]:
            if name:
                parts = name.strip().split(" ")
                if len(parts) >= 2:
                    last = parts[-1]
                    first_parts = parts[:-1]
                    initials = "".join(
                        (p[0].upper() if p else "undefined") + "." for p in first_parts)
                    formatted.append(f"{last} {initials}")
                else:
                    formatted.append(name)
        if len(authors) > 6:
            if len(formatted) >= 3:
                return ", ".join(formatted[:3]) + ", et al"
            return ", ".join(formatted) + ", et al"
        return ", ".join(formatted)
    except Exception:
        return "Unknown authors"


class LiteratureSearchStore:
    def __init__(self, openai_client, pg_dsn=DEFAULT_PG_DSN, probes=IVFFLAT_PROBES, verbose=True):
        self.client = openai_client
        self.pg_dsn = pg_dsn
        self.probes = probes
        self.verbose = verbose
        self.conn = None
        self._connect()

    # ------------------------------------------------------------------ connect
    def _log(self, msg):
        if self.verbose:
            print(msg, file=sys.stderr, flush=True)

    def _connect(self):
        try:
            self.conn = psycopg2.connect(self.pg_dsn)
        except Exception as e:
            raise RuntimeError(
                f"Could not connect to the literature database ({self.pg_dsn!r}): {e}\n"
                "Set it up and load it first — see the README, then run "
                "tools/build_literature_index/04_load_postgres.py.")
        cur = self.conn.cursor()
        # Same query-time setting the production API applies before every search.
        cur.execute(f"SET ivfflat.probes = {int(self.probes)}")
        cur.execute("SELECT to_regclass('works_vec'), to_regclass('works_meta_large')")
        vec, meta = cur.fetchone()
        if vec is None or meta is None:
            raise RuntimeError(
                "The literature database is missing the works_vec / works_meta_large tables. "
                "Load it with tools/build_literature_index/04_load_postgres.py.")
        self.conn.commit()
        if self.verbose:
            cur.execute("SELECT count(*) FROM works_vec")
            n = cur.fetchone()[0]
            self._log(f"Connected to literature index ({n:,} works, ivfflat.probes={self.probes}).")

    # ------------------------------------------------------------------ query
    def _embed_query(self, query):
        """Embed a query as production does: 'query: ' prefix, 1536 dims. Returns the
        pgvector input string (JSON array, the same serialization the API sends)."""
        resp = call_with_retry(self.client.embeddings.create,
                               model=EMBED_MODEL, input="query: " + query, dimensions=EMBED_DIM)
        return json.dumps(resp.data[0].embedding, separators=(",", ":"))

    def _build_base_query(self, has_abstract, qvec, year_from, year_to,
                          citations_min, citations_max, journals):
        """Port of the production controller's buildBaseQuery: same filters, same LIMIT 5."""
        sql = BASE_SELECT
        params = {"qvec": qvec, "has_abstract": has_abstract}
        sql += " AND m.has_abstract = %(has_abstract)s"

        if journals:
            params["journals"] = (journals if isinstance(journals, (list, tuple))
                                  else str(journals).split(","))
            sql += " AND m.journal = ANY(%(journals)s)"

        if year_from and year_to:
            params["year_from"], params["year_to"] = year_from, year_to
            sql += " AND m.year BETWEEN %(year_from)s AND %(year_to)s"
        elif year_from:
            params["year_from"] = year_from
            sql += " AND m.year >= %(year_from)s"
        elif year_to:
            params["year_to"] = year_to
            sql += " AND m.year <= %(year_to)s"

        if citations_min and citations_max:
            params["cmin"], params["cmax"] = int(citations_min), int(citations_max)
            sql += " AND COALESCE(m.cited_by_count, 0) BETWEEN %(cmin)s AND %(cmax)s"
        elif citations_min:
            params["cmin"] = int(citations_min)
            sql += " AND COALESCE(m.cited_by_count, 0) >= %(cmin)s"
        elif citations_max:
            params["cmax"] = int(citations_max)
            sql += " AND COALESCE(m.cited_by_count, 0) <= %(cmax)s"

        sql += " ORDER BY v.embedding <=> %(qvec)s LIMIT 5"
        return sql, params

    def _run(self, cur, has_abstract, qvec, year_from, year_to,
             citations_min, citations_max, journals):
        sql, params = self._build_base_query(
            has_abstract, qvec, year_from, year_to, citations_min, citations_max, journals)
        cur.execute(sql, params)
        return cur.fetchall()

    def search(self, query, year_from=None, year_to=None, citations_min=None,
               citations_max=None, journals=None, need_abstract=True):
        """Search the literature index, reproducing CaBot's /api/search semantics exactly.

        need_abstract=True  -> top-5 works WITH an abstract.
        need_abstract=False -> top-5 works WITHOUT an abstract, then top-5 WITH one
                               (the order the simple-line literature mode expects).

        Returns ``{"query", "results": [...], "totalResults"}`` where each result has
        id, title, authors (str), journal, year, abstract, citationCount, doi, score.
        """
        qvec = self._embed_query(query)
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            if need_abstract:
                rows = self._run(cur, True, qvec, year_from, year_to,
                                 citations_min, citations_max, journals)
            else:
                rows = (self._run(cur, False, qvec, year_from, year_to,
                                  citations_min, citations_max, journals)
                        + self._run(cur, True, qvec, year_from, year_to,
                                    citations_min, citations_max, journals))
            self.conn.commit()
        except Exception:
            self.conn.rollback()
            raise
        results = [self._row(r) for r in rows]
        return {"query": query, "results": results, "totalResults": len(results)}

    def _row(self, r):
        authors = list(r["authors"]) if r.get("authors") else []
        return {
            "id": r["id"],
            "title": r["title"] or "",
            "authors": format_authors_nejm(authors),
            "journal": r["journal"] or "",
            "year": r["year"],
            "abstract": r["abstract"] or "",
            "citationCount": r["cited_by_count"] or 0,
            "doi": r["doi"] or "",
            "score": float(r["score"]) if r["score"] is not None else None,
        }

    def close(self):
        if self.conn is not None and not self.conn.closed:
            self.conn.close()
