#!/usr/bin/env python3
"""
test_search.py -- run sample clinical queries against the local literature index
and print the top results, so you can eyeball that retrieval is sensible.

Uses the EXACT production retrieval path (same as 05_search.py / the differential
agent's search_literature tool):
  * query embedded as "query: " + text  (text-embedding-3-small, 1536-d)
  * SET ivfflat.probes = 42
  * score = 1 - (embedding <=> qvec)     (cosine via the IVFFlat index)
  * needAbstract=true: only papers that have an abstract are returned (this is the
    subset the differential models actually search).

Requirements: the literature index loaded into PostgreSQL + pgvector (fetch_data.py)
and OPENAI_API_KEY in config.ini. Reads PG_DSN the same way the rest of CaBot does.

Usage:
    .venv/bin/python test_search.py                       # built-in sample queries
    .venv/bin/python test_search.py --query "GLP-1 cardiovascular outcomes"
    .venv/bin/python test_search.py --query "X" --query "Y" --k 10
    .venv/bin/python test_search.py --dsn "dbname=cabot_search"
"""
import argparse
import json
import sys
from pathlib import Path

import psycopg2
import psycopg2.extras

# Reuse CaBot's own embedding/DSN helpers so this matches production exactly.
TOOLS = Path(__file__).resolve().parent / "tools" / "build_literature_index"
sys.path.insert(0, str(TOOLS))
from common import (  # noqa: E402
    EMBED_DIM, IVFFLAT_PROBES, build_query_text, get_embedding_client, get_pg_dsn)

DEFAULT_QUERIES = [
    "pheochromocytoma episodic hypertension elevated plasma metanephrines",
    "SGLT2 inhibitors heart failure with preserved ejection fraction outcomes",
    "tuberculous meningitis cerebrospinal fluid findings and treatment",
    "anti-NMDA receptor encephalitis presenting with psychosis and seizures",
    "hereditary hemochromatosis HFE mutation iron overload management",
    "immune checkpoint inhibitor associated colitis diagnosis and management",
    "acute promyelocytic leukemia treatment ATRA and arsenic trioxide",
    "antiphospholipid syndrome thrombosis and recurrent pregnancy loss",
    "giant cell arteritis temporal artery biopsy and tocilizumab",
    "pulmonary alveolar proteinosis whole lung lavage GM-CSF autoantibody",
]

# needAbstract=true: restrict to abstract-bearing works (the differential search space).
SQL = """
    SELECT m.id, m.title, 1 - (v.embedding <=> %(qvec)s) AS score,
           m.year, m.journal, m.cited_by_count, m.abstract
    FROM   works_meta_large AS m
    JOIN   works_vec AS v USING (id)
    WHERE  m.has_abstract = TRUE
    ORDER BY v.embedding <=> %(qvec)s
    LIMIT  %(k)s
"""


def parse_args():
    p = argparse.ArgumentParser(
        description="Run sample clinical queries against the literature index (needAbstract=true).")
    p.add_argument("--query", "-q", action="append", dest="queries",
                   help="A query to run (repeatable). Default: the built-in sample set.")
    p.add_argument("--k", type=int, default=5, help="Results per query (default: 5).")
    p.add_argument("--dsn", default=None,
                   help="libpq DSN (default: config.ini [main] PG_DSN / env PG_DSN / local).")
    return p.parse_args()


def main():
    args = parse_args()
    queries = args.queries or DEFAULT_QUERIES

    try:
        client, model_id = get_embedding_client()
    except Exception as e:
        sys.exit(f"OpenAI client error: {e}\nSet OPENAI_API_KEY in config.ini.")

    dsn = args.dsn or get_pg_dsn()
    try:
        conn = psycopg2.connect(dsn)
    except Exception as e:
        sys.exit(f"Could not connect to Postgres (DSN: {dsn!r}): {e}\n"
                 "Is the literature index loaded? See fetch_data.py / the README.")
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(f"SET ivfflat.probes = {IVFFLAT_PROBES}")

    for i, q in enumerate(queries, 1):
        resp = client.embeddings.create(
            model=model_id, input=build_query_text(q), dimensions=EMBED_DIM)
        qvec = json.dumps(resp.data[0].embedding, separators=(",", ":"))
        cur.execute(SQL, {"qvec": qvec, "k": args.k})
        rows = cur.fetchall()

        print(f"\n{'=' * 100}\n[{i:>2}] {q}\n{'=' * 100}")
        if not rows:
            print("   (no results)")
            continue
        for rank, r in enumerate(rows, 1):
            print(f"{rank}. [{r['score']:.3f}] {r['title']}")
            print(f"      {r['journal']} ({r['year']}) · {r['cited_by_count']} cites")
        snippet = (rows[0]["abstract"] or "").replace("\n", " ")[:320]
        print(f"   -- #1 abstract: {snippet}{'...' if len(rows[0]['abstract'] or '') > 320 else ''}")

    conn.close()
    print(f"\n{'=' * 100}\nDone: {len(queries)} queries.")


if __name__ == "__main__":
    main()
