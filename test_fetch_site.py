#!/usr/bin/env python3
"""
Test just the exemplar-index download — the `fetch_site` device flow from
fetch_data.py — without touching PostgreSQL or the literature index.

It exercises the CPC-Bench download path end to end: POST /api/device/start →
browser approval at {site}/activate → poll /api/device/poll → download
cpc_presentation_index_100.parquet from /api/dataset. The file lands in --data-dir.

    python test_fetch_site.py                                # https://cpcbench.com
    python test_fetch_site.py --site http://localhost:3008   # a local dev server
    python test_fetch_site.py --data-dir /tmp/cabot-test
"""
import argparse

from fetch_data import fetch_site, DEFAULT_SITE


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", default=DEFAULT_SITE, help="CPC-Bench site base URL")
    ap.add_argument("--data-dir", default="data", help="Where to write the exemplar parquet")
    args = ap.parse_args()
    fetch_site(args.site, args.data_dir)
    print("\nfetch_site OK.")


if __name__ == "__main__":
    main()
