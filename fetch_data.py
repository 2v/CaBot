#!/usr/bin/env python3
"""
fetch_data.py -- set up everything CaBot needs to run.

Two datasets:

  1. The literature-search index -> local PostgreSQL + pgvector. This downloads the
     index from HuggingFace (tbuckley/cabot-search) and loads it into your local
     database with the exact production schema and IVFFlat index, by running
     tools/build_literature_index/04_load_postgres.py. Needed by every version.
     (Requires PostgreSQL + pgvector already installed and a database created — see
     the README "Setup" section. The full load is ~3.47M rows + an index build:
     budget ~1 hour. Use --max-rows for a quick subset.)

  2. The 100-public-CPC exemplar index (cpc_presentation_index_100.parquet, ~1.2 MB)
     from the CPC-Bench site, written to data/. Needed only by the differential-
     diagnosis versions that use exemplar retrieval (v1, v1.1). This file is gated
     behind a free registration, so the download uses a one-time browser approval:
     the script prints a code, opens your browser to {site}/activate, you sign in
     and approve, and the file downloads.

Usage:
    python fetch_data.py                         # both
    python fetch_data.py --skip-site             # literature DB only (vr1 / vs*)
    python fetch_data.py --skip-postgres         # exemplar index only
    python fetch_data.py --max-rows 200000       # load only a subset of the literature index
    python fetch_data.py --dsn "dbname=cabot_search host=localhost port=5432"
"""
import argparse
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

import requests

DEFAULT_SITE = "https://cpcbench.com"
DATASET_FILE = "cpc_presentation_index_100.parquet"
LOAD_SCRIPT = Path(__file__).resolve().parent / "tools" / "build_literature_index" / "04_load_postgres.py"

# Private/gated full CPC exemplar index (the >6,000-case corpus CaBot v1 retrieved over).
FULL_CPC_REPO = "tbuckley/cabot-cpc-index-full"
FULL_CPC_FILE = "cpc_presentation_index_full.parquet"


def parse_args():
    p = argparse.ArgumentParser(
        description="Load CaBot's literature index into PostgreSQL + pgvector and pull the CPC "
                    "exemplar index (CPC-Bench site, one-time browser approval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--site", default=DEFAULT_SITE, help="CPC-Bench site base URL")
    p.add_argument("--data-dir", default="data", help="Where to write the exemplar parquet")
    p.add_argument("--dsn", default=None,
                   help="libpq DSN for the literature database (default: config.ini / env / local)")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Load only this many literature rows (faster; retrieval then "
                        "searches just that subset)")
    p.add_argument("--skip-postgres", action="store_true", help="Skip the literature DB load")
    p.add_argument("--skip-site", action="store_true", help="Skip the CPC-Bench exemplar index")
    p.add_argument("--full-cpc-index", action="store_true",
                   help="Also pull the PRIVATE full CPC exemplar index from HuggingFace "
                        f"({FULL_CPC_REPO}). Requires an HF token with access (config.ini "
                        "[main] HF_WRITE_TOKEN, env HF_TOKEN, or `huggingface-cli login`). "
                        "When present, CaBot-Public uses it automatically for v1 / v1.1 "
                        "exemplar retrieval; otherwise it falls back to the 100 public CPCs.")
    p.add_argument("--full-cpc-repo", default=FULL_CPC_REPO,
                   help="HuggingFace dataset repo for the full CPC index")
    p.add_argument("--agree-terms", action="store_true",
                   help="Affirm that you have read and agree to the CPC-Bench Terms of Use "
                        "(skips the interactive prompt; for non-interactive runs)")
    return p.parse_args()


def confirm_terms(site, agreed_via_flag):
    """Require the user to affirm the Terms of Use before the download flow starts."""
    terms_url = f"{site}/terms-of-use.html"
    if agreed_via_flag:
        print(f"      Terms of Use accepted via --agree-terms ({terms_url}).")
        return
    print("      ------------------------------------------------------------------")
    print("      This dataset is provided under the CPC-Bench Terms of Use:")
    print(f"        {terms_url}")
    print("      Please review them before downloading.")
    print("      ------------------------------------------------------------------")
    try:
        answer = input('      Type "agree" to confirm you have read and agree '
                       'to the Terms of Use: ').strip().lower()
    except EOFError:
        sys.exit("\n      No interactive terminal available. Re-run with --agree-terms "
                 "after reading the Terms of Use.")
    if answer not in ("agree", "i agree", '"agree"'):
        sys.exit("      Download cancelled: you must agree to the Terms of Use.")


def fetch_postgres(args):
    print("\n[1/2] Literature index -> PostgreSQL + pgvector")
    if not LOAD_SCRIPT.exists():
        print(f"      Load script not found at {LOAD_SCRIPT}; skipping.")
        return
    cmd = [sys.executable, str(LOAD_SCRIPT)]
    if args.dsn:
        cmd += ["--dsn", args.dsn]
    if args.max_rows:
        cmd += ["--max-rows", str(args.max_rows)]
    print("      Downloading the index from HuggingFace and loading it into Postgres")
    print("      (one-time; the full load is ~3.47M rows + an IVFFlat build — budget ~1 hour).")
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n      The Postgres load failed. Make sure PostgreSQL + pgvector are installed")
        print("      and the database exists (see the README 'Setup' section), then re-run:")
        print(f"        python {LOAD_SCRIPT}")


def fetch_site(site, data_dir, agree_terms=False):
    site = site.rstrip("/")
    print(f"\n[2/2] Exemplar index: requesting access from {site} ...")

    # 1. confirm the Terms of Use, then start a device-authorization flow
    confirm_terms(site, agree_terms)
    try:
        r = requests.post(f"{site}/api/device/start",
                          json={"terms_accepted": True}, timeout=30)
        r.raise_for_status()
        start = r.json()
    except Exception as e:
        sys.exit(f"      Could not start the download flow: {e}")

    device_code = start["device_code"]
    user_code = start["user_code"]
    verification_url = start.get("verification_url", f"{site}/activate")
    interval = int(start.get("interval", 3))
    expires_in = int(start.get("expires_in", 600))

    activate_url = f"{verification_url}?code={user_code}"
    print("      ------------------------------------------------------------------")
    print(f"      To authorize this download, open:\n        {verification_url}")
    print(f"      and enter this code:  {user_code}")
    print("      (Sign in with Google the first time, then click Approve.)")
    print("      ------------------------------------------------------------------")
    try:
        webbrowser.open(activate_url)
    except Exception:
        pass

    # 2. poll until approved / denied / expired
    deadline = time.monotonic() + expires_in
    token = None
    print("      Waiting for approval", end="", flush=True)
    while time.monotonic() < deadline:
        time.sleep(interval)
        print(".", end="", flush=True)
        try:
            pr = requests.post(f"{site}/api/device/poll",
                               json={"device_code": device_code}, timeout=30)
            data = pr.json()
        except Exception:
            continue
        status = data.get("status")
        if status == "approved":
            token = data.get("token")
            break
        if status == "denied":
            sys.exit("\n      Access was denied on the website.")
        if status in ("expired", "consumed"):
            sys.exit("\n      The code expired before approval. Re-run fetch_data.py.")
    print()
    if not token:
        sys.exit("      Timed out waiting for approval. Re-run fetch_data.py.")

    # 3. download the file with the one-time bearer token
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / DATASET_FILE
    print(f"      Approved. Downloading {DATASET_FILE} ...")
    try:
        with requests.get(f"{site}/api/dataset/{DATASET_FILE}",
                          headers={"Authorization": f"Bearer {token}"},
                          stream=True, timeout=120) as dr:
            dr.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in dr.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
    except Exception as e:
        sys.exit(f"      Download failed: {e}")
    print(f"      Exemplar index saved to: {dest}")


def _optional_hf_token():
    """Best-effort HF token for the private index: config.ini / env, else None
    (snapshot_download also honors a prior `huggingface-cli login`)."""
    try:
        sys.path.insert(0, str(LOAD_SCRIPT.parent))
        from common import get_hf_token
        return get_hf_token()
    except Exception:
        return None


def fetch_full_cpc_index(repo_id, data_dir):
    """Pull the private full CPC exemplar index from HuggingFace into data/.

    Optional and gated: needs an HF token with access to the repo. On any failure
    (no token / no access / offline) it prints how to authenticate and returns
    without error, so CaBot-Public falls back to the 100 public CPCs.
    """
    print(f"\n[full] Private full CPC exemplar index <- HuggingFace ({repo_id})")
    try:
        from huggingface_hub import snapshot_download
    except Exception:
        print("      huggingface_hub not installed; skipping. `pip install huggingface_hub`.")
        return
    token = _optional_hf_token()
    try:
        path = snapshot_download(repo_id=repo_id, repo_type="dataset",
                                 allow_patterns=[FULL_CPC_FILE], token=token)
    except Exception as e:
        print(f"      Could not download the full CPC index: {e}")
        print("      This dataset is private — authenticate with an account that has access:")
        print("        set [main] HF_WRITE_TOKEN in config.ini, export HF_TOKEN, or run")
        print("        `huggingface-cli login`. CaBot-Public will use the 100 public CPCs "
              "until then.")
        return
    src = Path(path) / FULL_CPC_FILE
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    dest = data_dir / FULL_CPC_FILE
    # Copy out of the HF cache so run_cabot finds it at the default data/ path.
    import shutil
    shutil.copyfile(src, dest)
    size_mb = dest.stat().st_size / 1e6
    print(f"      Full CPC index saved to: {dest} ({size_mb:.1f} MB)")
    print("      v1 / v1.1 will now retrieve exemplars over the full corpus automatically.")


def main():
    args = parse_args()
    if args.skip_postgres and args.skip_site and not args.full_cpc_index:
        sys.exit("Nothing to do: both --skip-postgres and --skip-site were given.")
    if not args.skip_postgres:
        fetch_postgres(args)
    if not args.skip_site:
        fetch_site(args.site, args.data_dir, agree_terms=args.agree_terms)
    if args.full_cpc_index:
        fetch_full_cpc_index(args.full_cpc_repo, args.data_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
