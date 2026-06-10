#!/usr/bin/env python3
"""
05_search.py -- search the local pgvector database exactly as the production
CaBot /api/search endpoint does.

This is the QUICKSTART entry point (after loading the database with
04_load_postgres.py). It runs the SAME SQL as the production Node controller,
against the same schema, with the same ANN configuration:

  * query embedded as "query: " + text (text-embedding-3-small, 1536-d);
  * SET ivfflat.probes = 42 on the session (production sets the same value);
  * score = 1 - (embedding <=> query)  -- cosine similarity via the IVFFlat
    index (lists = 1732, vector_cosine_ops);
  * the same WHERE clauses for yearFrom/yearTo/citationsMin/citationsMax/
    journals, and the same dual retrieval: needAbstract=false returns the top 5
    works WITHOUT an abstract followed by the top 5 WITH one; needAbstract=true
    returns only the top 5 with an abstract.

Two output modes:
  * default  -- a compact human-readable top-k ranking (single query, no
                abstract split; purely for eyeballing results).
  * --json   -- the EXACT CaBot /api/search JSON response (same fields, key
                order, NEJM author/citation formatting, filters).

Reproducibility: the method is identical to production. Two caveats are
inherent to IVFFlat and worth knowing when comparing against the original
server: (1) index builds train centroids with randomized k-means, so a rebuilt
index is a different instance of the same method and the approximate top-5 can
occasionally differ from the original server's specific index; (2) production
sets ivfflat.probes through a connection pool, so the very first requests after
a server restart could have run on connections where the SET had not yet been
applied -- here the SET is guaranteed on the session, which is the intended
configuration.

Usage:
    python 05_search.py                                  # built-in sample queries
    python 05_search.py --query "GLP-1 cardiovascular outcomes" --k 10
    # Exact CaBot API response as JSON:
    python 05_search.py --query "CAR-T relapsed B-cell lymphoma" --json
    python 05_search.py --query "statins" --json --need-abstract \\
        --year-from 2015 --year-to 2023 --citations-min 50
"""

import argparse
import json
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).resolve().parent))
import apiformat  # noqa: E402
from common import (EMBED_DIM, IVFFLAT_PROBES, build_query_text,  # noqa: E402
                    get_embedding_client, get_pg_dsn)

SAMPLE_QUERIES = [
    "GLP-1 receptor agonists and cardiovascular outcomes in type 2 diabetes",
    "CAR-T cell therapy for relapsed large B-cell lymphoma",
    "mRNA vaccine efficacy against severe COVID-19",
]

# The SELECT list, join, and ordering below are a literal transcription of the
# production controller (ooe_site/server/src/controllers/search.js).
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


def log(msg):
    """Progress/info goes to stderr so --json output on stdout stays clean."""
    print(msg, file=sys.stderr)


def embed_query(query):
    """Embed a query as production does: 'query: ' prefix, 1536 dims. Returns the
    pgvector input string (JSON array, same serialization the API sends)."""
    client, model_id = get_embedding_client()
    resp = client.embeddings.create(
        model=model_id, input=build_query_text(query), dimensions=EMBED_DIM)
    return json.dumps(resp.data[0].embedding, separators=(",", ":"))


def build_base_query(has_abstract, qvec_str, args):
    """Port of the controller's buildBaseQuery: same filters, same LIMIT 5."""
    sql = BASE_SELECT
    params = {"qvec": qvec_str, "has_abstract": has_abstract}
    sql += " AND m.has_abstract = %(has_abstract)s"

    if args.journals:
        params["journals"] = args.journals.split(",")
        sql += " AND m.journal = ANY(%(journals)s)"

    if args.year_from and args.year_to:
        params["year_from"], params["year_to"] = args.year_from, args.year_to
        sql += " AND m.year BETWEEN %(year_from)s AND %(year_to)s"
    elif args.year_from:
        params["year_from"] = args.year_from
        sql += " AND m.year >= %(year_from)s"
    elif args.year_to:
        params["year_to"] = args.year_to
        sql += " AND m.year <= %(year_to)s"

    if args.citations_min and args.citations_max:
        params["cmin"], params["cmax"] = int(args.citations_min), int(args.citations_max)
        sql += " AND COALESCE(m.cited_by_count, 0) BETWEEN %(cmin)s AND %(cmax)s"
    elif args.citations_min:
        params["cmin"] = int(args.citations_min)
        sql += " AND COALESCE(m.cited_by_count, 0) >= %(cmin)s"
    elif args.citations_max:
        params["cmax"] = int(args.citations_max)
        sql += " AND COALESCE(m.cited_by_count, 0) <= %(cmax)s"

    sql += " ORDER BY v.embedding <=> %(qvec)s LIMIT 5"
    return sql, params


def search_api(cur, qvec_str, args):
    """The production retrieval: needAbstract=true -> top 5 with abstracts;
    otherwise top 5 without an abstract followed by top 5 with one."""
    def run(has_abstract):
        sql, params = build_base_query(has_abstract, qvec_str, args)
        cur.execute(sql, params)
        return cur.fetchall()

    if args.need_abstract:
        return run(True)
    return run(False) + run(True)


def search_human(cur, qvec_str, k):
    cur.execute("""
        SELECT m.id, m.title, 1 - (v.embedding <=> %(qvec)s) AS score,
               m.year, m.journal, m.cited_by_count, m.doi, m.is_open_access
        FROM   works_meta_large AS m
        JOIN   works_vec  AS v USING (id)
        ORDER BY v.embedding <=> %(qvec)s
        LIMIT %(k)s
    """, {"qvec": qvec_str, "k": k})
    return cur.fetchall()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dsn", help="libpq DSN (default: config.ini [main] PG_DSN / env PG_DSN).")
    ap.add_argument("--query", help="A single query (default: built-in samples).")
    ap.add_argument("--k", type=int, default=5, help="Results per query, human mode (default: 5).")
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

    dsn = args.dsn or get_pg_dsn()
    conn = psycopg2.connect(dsn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    # Same query-time setting the production API applies before every search.
    cur.execute(f"SET ivfflat.probes = {IVFFLAT_PROBES}")

    if args.json:
        query = args.query or SAMPLE_QUERIES[0]
        qvec_str = embed_query(query)
        rows = search_api(cur, qvec_str, args)
        resp = apiformat.build_response(query, rows, user_api_key_present=True)
        print(json.dumps(resp, indent=2, ensure_ascii=False))
        return

    queries = [args.query] if args.query else SAMPLE_QUERIES
    for query in queries:
        qvec_str = embed_query(query)
        rows = search_human(cur, qvec_str, args.k)
        print(f"\n=== {query}")
        for rank, r in enumerate(rows, 1):
            oa = "OA" if r["is_open_access"] else "  "
            print(f"{rank:>2}. [{r['score']:.3f}] {oa} {r['title']}")
            print(f"      {r['journal']} ({r['year']}) · "
                  f"{r['cited_by_count']} cites · {r['doi']}")
    conn.close()


if __name__ == "__main__":
    main()
