#!/usr/bin/env python3
"""
CaBot-Public command-line runner.

Generates a text differential diagnosis and/or a video slideshow presentation for a single
case presentation supplied as a plain-text file.

Examples
--------
  # newest model, text + video (default). The bundled example is NEJM Case 5-2025
  # (NEJMcpc2412514), one of the 100 public CPC exemplars — exclude it from retrieval:
  python run_cabot.py --case examples/example_case.txt --output out/ --exclude-id NEJMcpc2412514

  # text only, original A/B-test model
  python run_cabot.py --case examples/example_case.txt --output out/ --version v1 --mode text \
      --exclude-id NEJMcpc2412514

  # rare-disease (UDN) model, text only (its canonical mode)
  python run_cabot.py --case examples/udn_example.txt --output out/ --version vr1 --mode text

  # video only, overriding the version's default base model
  python run_cabot.py --case examples/example_case.txt --output out/ \
      --version v1.1 --mode video --base-model gpt-5 --exclude-id NEJMcpc2412514

See README.md for the full version / mode matrix and setup.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from configparser import ConfigParser

from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cabot_public_lib.versions import get_version, VERSIONS, DEFAULT_VERSION
from cabot_public_lib.cabot import CaBot, DEFAULT_NEJM_CPCS_PATH, resolve_cpc_index
from cabot_public_lib.literature_store import LiteratureSearchStore, DEFAULT_PG_DSN
from cabot_public_lib.cpc_presentation_store import CPCPresentationStore


def parse_args():
    p = argparse.ArgumentParser(
        description="Run CaBot on a case presentation (text differential + video slideshow).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--case", "-c", required=True, help="Path to a .txt file with the case presentation")
    p.add_argument("--output", "-o", required=True, help="Output directory")
    p.add_argument("--version", "-v", default=DEFAULT_VERSION, choices=list(VERSIONS),
                   help="Model version")
    p.add_argument("--mode", "-m", default="both", choices=["text", "video", "both"],
                   help="What to generate")
    p.add_argument("--base-model", default=None,
                   help="Override the version's default base model (e.g. o3, gpt-5, gpt-5.4)")
    p.add_argument("--max-iterations", type=int, default=None,
                   help="Override the version's iteration budget (tool calls + responses). "
                        "Versions whose prompt states the budget (v1.1, vr1, vs*) tell the "
                        "model the new number; v1's prompt does not, so the run is simply "
                        "cut off at the limit.")
    p.add_argument("--exclude-id", default=None,
                   help="Case ID or DOI to exclude from exemplar retrieval and literature citations "
                        "(e.g. when running a known NEJM case). Searched against the case database.")
    p.add_argument("--exclude-title", default=None,
                   help="Case title to exclude; resolved to a case ID against the case database.")
    p.add_argument("--config", default="config.ini", help="Path to config.ini with API keys")
    p.add_argument("--cpc-index", default=None,
                   help="Explicit parquet index for exemplar retrieval (v1, v1.1). Overrides "
                        "--cpc-index-set; default: resolved by --cpc-index-set.")
    p.add_argument("--cpc-index-set", choices=["auto", "full", "public"], default="auto",
                   help="Which exemplar index to use: 'public' (100 public CPCs), 'full' (the "
                        "private >6,000-case corpus v1 used, if pulled via fetch_data.py "
                        "--full-cpc-index), or 'auto' (prefer full when present, else public).")
    p.add_argument("--year-anchor", type=int, default=None,
                   help="Anchor year for exemplar retrieval: exemplars are restricted to "
                        "anchor-2 .. anchor+2 (exactly as v1 ran). Default: the version's anchor "
                        "(2022 for v1/v1.1). For a faithful run of a known dated case against the "
                        "full corpus, pass the case's publication year (e.g. --year-anchor 2025).")
    p.add_argument("--nejm-cpcs-path", default=DEFAULT_NEJM_CPCS_PATH,
                   help="Dir with case images for video generation (mode video/both)")
    p.add_argument("--pg-dsn", default=None,
                   help="libpq DSN for the local pgvector literature database "
                        "(default: config.ini [main] PG_DSN / env PG_DSN / "
                        f"'{DEFAULT_PG_DSN}'). Load it with "
                        "tools/build_literature_index/04_load_postgres.py.")
    p.add_argument("--debug", action="store_true", help="Verbose model I/O")
    return p.parse_args()


def load_keys(config_path, cli_pg_dsn=None):
    cfg = ConfigParser()
    if not os.path.exists(config_path):
        sys.exit(f"Error: config file '{config_path}' not found. Copy config.example.ini to "
                 f"config.ini and fill in your keys.")
    cfg.read(config_path)
    def get(*names):
        for n in names:
            if cfg.has_option("main", n):
                return cfg.get("main", n)
        return None
    api_key = get("OPENAI_API_KEY", "OPENAI_KEY_TB", "OPENAI_KEY")
    if not api_key:
        sys.exit("Error: OPENAI_API_KEY missing from [main] in config.ini")
    # libpq DSN for the pgvector literature database: --pg-dsn, else [main] PG_DSN,
    # else env PG_DSN, else the local default.
    pg_dsn = (cli_pg_dsn or get("PG_DSN") or os.environ.get("PG_DSN", "").strip()
              or DEFAULT_PG_DSN)
    return api_key, pg_dsn


def presentation_cfg_from_version(v):
    """Translate a VersionConfig's presentation fields into the dict video.py expects."""
    if v.presentation_style == "monolithic":
        return {"style": "monolithic", "prompt": v.presentation_prompt}
    return {
        "style": "split",
        "with_ddx_prefix": v.presentation_with_ddx_prefix,
        "without_ddx_prefix": v.presentation_without_ddx_prefix,
        "body": v.presentation_body,
    }


def main():
    args = parse_args()
    if not os.path.exists(args.case):
        sys.exit(f"Error: case file '{args.case}' not found")

    api_key, pg_dsn = load_keys(args.config, args.pg_dsn)
    v = get_version(args.version)
    if v.mode == "simple_qa" and args.mode == "video":
        sys.exit(f"Error: version '{v.name}' is a simple QA model and does not support video.")
    case_text = Path(args.case).read_text().strip()
    case_id = Path(args.case).stem
    out_dir = Path(args.output) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    client = OpenAI(api_key=api_key)

    # Resolve which exemplar index this run will use (only matters for v1 / v1.1).
    cpc_index_path, cpc_index_which = resolve_cpc_index(args.cpc_index, args.cpc_index_set)
    if v.use_similar_cases:
        label = {"explicit": "explicit", "full": "FULL private corpus",
                 "public": "100 public CPCs"}[cpc_index_which]
        print(f"Exemplar index: {cpc_index_path}  [{label}]")

    def build_stores():
        """Build CaBot's retrieval stores (both connect/load in their constructors):

          - literature_store: connects to the local pgvector database (needed by every
            version; load it once with tools/build_literature_index/04_load_postgres.py).
          - cpc_store: exemplar CPC retrieval, only for versions that use it (v1, v1.1).

        Video-only runs never reach here, so neither store is built for them.
        """
        literature_store = LiteratureSearchStore(client, pg_dsn=pg_dsn)
        cpc_store = (CPCPresentationStore(client, cpc_index_path=cpc_index_path)
                     if v.use_similar_cases else None)
        return literature_store, cpc_store

    # ---- simple QA / literature-search line (vs1, vs1.1): text-only, no exemplars, no video ----
    if v.mode == "simple_qa":
        print(f"=== CaBot {v.name} ({v.line} line) — simple QA / literature search — commit {v.repo_commit} ===")
        print(v.description)
        literature_store, cpc_store = build_stores()
        cabot = CaBot(
            client=client, version_config=v, literature_store=literature_store,
            cpc_store=cpc_store, nejm_cpcs_path=args.nejm_cpcs_path,
        )
        print("\nAnswering question (literature-grounded)...")
        qa_result = cabot.run_simple_literature(
            question=case_text, debug=args.debug, base_model=args.base_model,
            max_iterations=args.max_iterations,
        )
        answer_text = qa_result.get("output", "")
        record = {
            "id": case_id,
            "version": v.name,
            "repo_commit": v.repo_commit,
            "base_model": args.base_model or v.base_model,
            "mode": "simple_qa",
            "question": case_text,
            "cabot_prediction": qa_result,
            "prediction_timestamp": datetime.now().isoformat(),
        }
        (out_dir / "answer.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        (out_dir / "answer.md").write_text(answer_text)
        print(f"Saved answer -> {out_dir/'answer.json'}")
        print("\nDone.")
        return

    print(f"=== CaBot {v.name} ({v.line} line) — mode={args.mode} — commit {v.repo_commit} ===")
    print(v.description)

    ddx_text = ""
    ddx_result = None

    # ---- text differential ----
    if args.mode in ("text", "both"):
        literature_store, cpc_store = build_stores()
        cabot = CaBot(
            client=client, version_config=v, literature_store=literature_store,
            cpc_store=cpc_store, nejm_cpcs_path=args.nejm_cpcs_path,
        )
        print("\nGenerating differential diagnosis...")
        ddx_result = cabot.run(
            presentation_of_case=case_text,
            images=[],
            debug=args.debug,
            base_model=args.base_model,
            max_iterations=args.max_iterations,
            exclude_id=args.exclude_id,
            exclude_title=args.exclude_title,
            year_anchor=args.year_anchor,
        )
        ddx_text = ddx_result.get("output", "")

        record = {
            "id": case_id,
            "version": v.name,
            "repo_commit": v.repo_commit,
            "base_model": args.base_model or v.base_model,
            "mode": args.mode,
            "exclude_id": args.exclude_id,
            "exclude_title": args.exclude_title,
            "year_anchor": args.year_anchor if args.year_anchor is not None else v.year_anchor,
            "cpc_index": cpc_index_path if v.use_similar_cases else None,
            "presentation_of_case": case_text,
            "cabot_prediction": ddx_result,
            "prediction_timestamp": datetime.now().isoformat(),
        }
        (out_dir / "differential_diagnosis.json").write_text(json.dumps(record, indent=2, ensure_ascii=False))
        (out_dir / "differential_diagnosis.md").write_text(ddx_text)
        print(f"Saved differential diagnosis -> {out_dir/'differential_diagnosis.json'}")

    # ---- video slideshow ----
    if args.mode in ("video", "both"):
        from cabot_public_lib.video import generate_video_for_case
        video_base_model = args.base_model or v.default_video_base_model
        pres_cfg = presentation_cfg_from_version(v)
        case = {
            "id": case_id,
            "presentation_of_case": case_text,
            "presentation_of_case_references": [],  # CLI input is text-only; no figures
        }
        # video-only -> no DDx attached (WITHOUT_DDX presentation prefix)
        base_ddx = ddx_text if args.mode == "both" else ""
        print(f"\nGenerating video slideshow (model={video_base_model})...")
        _, success, error_msg = generate_video_for_case(
            case=case,
            base_differential_diagnosis=base_ddx,
            base_cpc_path=args.nejm_cpcs_path,
            api_key=api_key,
            output_base_dir=str(args.output),
            base_model=video_base_model,
            presentation_cfg=pres_cfg,
        )
        if success:
            print(f"Saved video -> {Path(args.output)/case_id/(case_id + '_presentation.mp4')}")
        else:
            print(f"Video generation failed: {error_msg}")

    print("\nDone.")


if __name__ == "__main__":
    main()
