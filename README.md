<div align="center">
  <img src="assets/cabot_transparent_small.png" alt="CaBot" width="150">
  <h1>CaBot</h1>
  <p><b>An AI expert discussant for clinical case conferences</b></p>
  <p>
    🌐 <a href="https://cpcbench.com">cpcbench.com</a> ·
    🤗 <a href="https://huggingface.co/datasets/tbuckley/cabot-search">literature index</a> ·
    <a href="https://cpcbench.com/ai-terms-of-use.html">terms of use</a>
  </p>
</div>

Dr. CaBot is a multimodal, interactive **AI expert discussant**: presented with a clinical case, it
responds with a written **differential diagnosis** in the style of the NEJM Clinicopathologic
Conference, along with an optional **video presentation** that narrates the reasoning.

This repository is the source code for CaBot. CaBot is an agentic system that equips a reasoning
model (OpenAI o3 or GPT-5.4, selected with `--version`) with external tools: it retrieves two
similar cases as exemplars of expert clinical reasoning, then iteratively searches 1,623,047
clinical abstracts to develop a differential diagnosis with citations (5 articles per query, ranked
by semantic similarity). With `--mode video`/`both`, CaBot generates a complete slideshow
presentation. Everything runs on your machine: a **self-hosted literature search** (PostgreSQL +
pgvector; embeddings published on HuggingFace,
[`tbuckley/cabot-search`](https://huggingface.co/datasets/tbuckley/cabot-search)) and **self-hosted
case retrieval** over 100 public NEJM CPC cases. The full literature index holds 3.47M articles
from 204 high-impact clinical journals — the differential models search its 1.6M abstract-bearing
subset, and the tool can search all 3.47M (see *Literature mode* under [Versions](#versions)).

> **Note:** the study's full exemplar corpus of over 6,000 clinical cases cannot be redistributed,
> so exemplar retrieval in this release searches the 100-case public
> [CPC-Bench](https://cpcbench.com) dataset (a year-stratified sample over 2000–2025) and can
> surface different exemplars than the original runs.

> **Terms of use:** by cloning, downloading, or otherwise using this repository, you agree to the
> [terms of use](https://cpcbench.com/ai-terms-of-use.html).

## Contents

- [Quick Start](#quick-start)
- [Versions](#versions)
- [Usage](#usage)
- [Rebuilding the literature index from OpenAlex](#rebuilding-the-literature-index-from-openalex)
- [Layout](#layout)
- [Citation](#citation)

## Quick Start

**Requirements.** Python 3.10+ (developed on 3.12) and PostgreSQL 17 with the pgvector extension
(installed in step 3 below). Linux and macOS are supported — the commands below are written for
Ubuntu/Debian; on macOS install Postgres via Homebrew (`brew install postgresql@17 pgvector`). On
Windows, use WSL2. Video generation (`--mode video`/`both`) additionally needs `pdflatex` (TeX
Live/MacTeX with the `beamer` class), `pdftoppm` (poppler), and `ffmpeg`/`ffprobe` on your PATH.

```bash
# 1. Python deps in a virtualenv
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. API key — copy the template and fill it in
cp config.example.ini config.ini   # set OPENAI_API_KEY in [main]

# 3. PostgreSQL + pgvector for the literature search (production: PostgreSQL 17.5,
#    pgvector 0.8.0). On Ubuntu/Debian, PostgreSQL 17 needs the official PGDG apt repo:
sudo apt install postgresql-common
sudo /usr/share/postgresql-common/pgdg/apt.postgresql.org.sh -y
sudo apt install postgresql-17 postgresql-17-pgvector
#    Make sure the server is running before the next steps. On systems without
#    systemd (e.g. Docker containers): sudo service postgresql start
sudo -u postgres createdb cabot_search
sudo -u postgres psql -d cabot_search -c "CREATE EXTENSION vector;"
sudo -u postgres createuser -s $USER   # Postgres role matching your OS user (peer auth)
sudo -u postgres psql -d cabot_search -c "GRANT ALL ON SCHEMA public TO $USER;"
#    (set PG_DSN in config.ini if these defaults don't fit — see config.example.ini)
#    The default DSN (host=localhost) connects over TCP, which asks for a password.
#    If you hit "fe_sendauth: no password supplied", connect via the unix socket
#    (peer auth, no password) instead:  export PG_DSN="dbname=cabot_search"

# 4. Data — load the literature index into Postgres + pull the exemplar index
#    (a gated download from https://cpcbench.com; one-time browser approval).
#    Optional: authenticate to HuggingFace for faster, non-rate-limited downloads
#    (any read token from https://huggingface.co/settings/tokens):
export HF_TOKEN=hf_...
python fetch_data.py
#    The full load is ~3.47M rows + an index build (budget ~1 hour). For quick
#    testing, add --max-rows 200000 to load a subset (retrieval then searches
#    just that subset; re-run without the flag later to load the rest).

# 5. Run an example case through CaBot (v1.1 = newest main-line version). The bundled
#    example is a short fictional teaching case (not from any case database), so no
#    --exclude-id is needed.
python run_cabot.py --case examples/example_case.txt --output out/ --version v1.1 --mode text

# 6. (Optional) Video generation — only needed for --mode video/both. Installs pdflatex with
#    the beamer class, pdftoppm (poppler), and ffmpeg/ffprobe. On Ubuntu/Debian:
sudo apt install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended \
    poppler-utils ffmpeg
#    (macOS: brew install --cask mactex-no-gui && brew install poppler ffmpeg)
```

## Versions

| Version | Line | Base model | Grounding | Literature mode | Exemplar window (year-anchor ±2) | Presentation | Notes |
|---------|------|-----------|-----------|-----------------|----------------------------------|--------------|-------|
| `v1`    | main | o3        | literature + exemplar CPC retrieval | abstracts | 2022 → 2020–2024 | standard | Model used in the physician A/B test |
| `v1.1`  | main (**default**) | gpt-5.4 | literature + exemplar CPC retrieval | abstracts | 2022 → 2020–2024 | acknowledges missing information | Newest main-line model |
| `vr1`   | rare | gpt-5.4   | literature only (no exemplars) | abstracts | — (no exemplars) | acknowledges missing information | Tuned for UDN rare-disease application letters; runs without local data |
| `vs1`   | simple | o3      | literature only (no exemplars) | abstracts + titles | — (no exemplars) | n/a (text only) | Simple QA / literature-search mode used for the NEJMBench QA & VQA benchmarks |
| `vs1.1` | simple | gpt-5.4 | literature only (no exemplars) | abstracts + titles | — (no exemplars) | n/a (text only) | Same as `vs1`, newer base model |

> **Exemplar window (v1 / v1.1).** Exemplar retrieval pulls the top-2 most similar CPCs from a
> 5-year window centered on a **year-anchor** (`anchor−2 … anchor+2`). The anchor defaults to
> **2022** (→ 2020–2024); override it per run with `--year-anchor`. The original v1 A/B run set the
> anchor to *each case's own publication year*, so to reproduce a known dated case faithfully, pass
> that year (e.g. `--year-anchor 2025`). The CPC exemplar corpus spans publication dates **through
> 2024-05-30** (cases on/before the o3 pretraining cutoff), so retrieval windows are effectively
> capped at that date. With the public 100-case index the window surfaces fewer / different
> exemplars than the paper; pull the full >6,000-case corpus with `fetch_data.py --full-cpc-index`
> (gated, requires HuggingFace access) to reproduce v1's full-corpus retrieval.


## Usage

```bash
# The bundled example case is a short fictional teaching case (not drawn from any case
# database), so the runs below need no --exclude-id. When you run a real published case
# that lives in the exemplar corpus, pass --exclude-id so it is never retrieved as its
# own exemplar and its source paper is never cited (see the final example below).

# Quickest way to try it: newest model, text only, no exemplar data needed (vr1).
# (Load a literature subset first for speed: fetch_data.py --skip-site --max-rows 200000)
python run_cabot.py --case examples/example_case.txt --output out/ --version vr1 --mode text

# Newest model, differential only (v1.1 needs the exemplar index from fetch_data.py)
python run_cabot.py --case examples/example_case.txt --output out/ --version v1.1 --mode text

# Newest model, full pipeline (differential + slideshow)
python run_cabot.py --case examples/example_case.txt --output out/

# Original A/B-test model, text only
python run_cabot.py --case examples/example_case.txt --output out/ --version v1 --mode text

# Simple QA / literature-search mode (the case file holds the question; o3, as benchmarked)
python run_cabot.py --case examples/example_question.txt --output out/ --version vs1

# Video-only demo
python run_cabot.py --case examples/example_case.txt --output out/ --version v1.1 --mode video --base-model gpt-5

# Run a KNOWN NEJM case while excluding it from exemplar retrieval and literature citations
python run_cabot.py --case case_4_2019.txt --output out/ --exclude-id NEJMcpc1810391
python run_cabot.py --case case.txt --output out/ --exclude-title "Case 4-2019: An 18-Year-Old Man with Fever"
```

When `--exclude-id`/`--exclude-title` is given, CaBot searches the case database by ID (titles are
resolved to an ID) and prints whether the case was found and will be excluded — so the identical
case is never retrieved as an exemplar and its source paper is never cited.

Outputs are written to `out/<case-name>/`:
- `differential_diagnosis.json` (full record) and `differential_diagnosis.md` (the text)
- `<case-name>_presentation.mp4` (when video is generated)

### Options

```
--case, -c          Path to a .txt file with the case presentation (required)
--output, -o        Output directory (required)
--version, -v       v1 | v1.1 | vr1 | vs1 | vs1.1   (default: v1.1)
--mode, -m          text | video | both   (default: both)
--base-model        Override the version's default base model (e.g. o3, gpt-5, gpt-5.4)
--max-iterations    Override the version's iteration budget (tool calls + responses). v1.1/vr1/vs*
                    state the budget in their prompt, so the model plans around the new number;
                    v1's prompt does not, so the run is simply cut off at the limit.
--exclude-id        Case ID to exclude from exemplar retrieval + literature citations
--exclude-title     Case title to exclude (resolved to a case ID against the case database)
--config            Path to config.ini (default: config.ini)
--cpc-index         Explicit parquet index for exemplar retrieval (overrides --cpc-index-set)
--cpc-index-set     auto | full | public  (default: auto — prefer the full private corpus when
                    it has been pulled, else the 100 public CPCs)
--year-anchor       Anchor year for exemplar retrieval (window = anchor-2 .. anchor+2). Default:
                    the version's anchor (2022 for v1/v1.1). Pass a case's publication year to
                    reproduce v1's per-case window against the full corpus.
--nejm-cpcs-path    Dir with case images for video generation (default: data/nejm_cpcs)
--pg-dsn            libpq DSN for the pgvector literature DB (default: config.ini / env / local)
--debug             Verbose model I/O
```


## Rebuilding the literature index from OpenAlex

The published index was built in **early June 2025** from an OpenAlex `works` snapshot dated
**~2025-06-05**. OpenAlex changes over time, so a rebuild from a newer snapshot yields a *similar but
not byte-identical* corpus — use a ~2025-06-05 snapshot to reproduce the paper's index exactly, or a
newer one to **refresh the index with recently published papers**. The full pipeline lives in
`tools/build_literature_index/` (raw OpenAlex → journal subset → embeddings → published dataset →
local Postgres). It uses the build dependencies already in `requirements.txt` (`tiktoken`, `orjson`,
`datasets`) and your `OPENAI_API_KEY` (a full run is a few billion embedding tokens — check pricing
first).

```bash
# 1. Download the OpenAlex works snapshot (free, no AWS account). Large (hundreds of GB);
#    we only need the works/ tree. For the paper's exact corpus, use a ~2025-06-05 snapshot.
aws s3 sync "s3://openalex/data/works" "openalex-snapshot/data/works" --no-sign-request

# 2. Filter the snapshot to the 204 high-impact journals (2023 JIF >= 10)
python tools/build_literature_index/01_filter_openalex.py \
    --snapshot openalex-snapshot/data/works
#   -> tools/build_literature_index/data/journal_works.jsonl.gz   (~3.5M works)

# 3. Embed each work with text-embedding-3-small (1536-d) -> sharded parquet
python tools/build_literature_index/02_build_embeddings.py
#   -> tools/build_literature_index/data/parquet/part-00000.parquet, ...

# 4. (optional) Publish the shards as your own HuggingFace dataset
python tools/build_literature_index/03_upload_huggingface.py --repo-id <your-username>/cabot-search

# 5. Load the shards into local PostgreSQL + pgvector
#    (same schema + IVFFlat index, lists=1732, as production)
python tools/build_literature_index/04_load_postgres.py \
    --local tools/build_literature_index/data/parquet --drop

# 6. Verify
python tools/build_literature_index/05_search.py --query "GLP-1 cardiovascular outcomes"
```

To test the pipeline end to end quickly, add `--limit 100000` to step 2 and `--max-rows 100000` to step 5.
Step 2 supports `--resume`; each script's `--help` / module docstring documents the rest (shard size,
embedding batch size, index-build tuning). If you publish your own index, point CaBot at it by passing
`--repo-id <your-username>/cabot-search` to `04_load_postgres.py` (and update the website download link).

## Layout

```
cabot_public_lib/
├── cabot_prompts.py          # every version's prompts, in full (edit these)
├── versions.py               # version registry + linking source commits
├── cabot.py                  # text differential engine
├── video.py                  # slideshow pipeline (LaTeX -> PDF -> images -> TTS -> mp4)
├── cpc_presentation_store.py # exemplar CPC retrieval (100 public CPCs, in-memory numpy)
└── literature_store.py       # literature search (PostgreSQL + pgvector client)
run_cabot.py                  # CLI
fetch_data.py                 # loads the literature index into Postgres + pulls the exemplar index
tools/build_literature_index/ # rebuild the index from OpenAlex; 04 loads Postgres, 05 searches
```

## Citation

**Teaching large language models to reason like expert diagnosticians**
([arXiv:2509.12194](https://arxiv.org/abs/2509.12194))

```bibtex
@misc{buckley2026teachinglargelanguagemodels,
      title={Teaching large language models to reason like expert diagnosticians},
      author={Thomas A. Buckley and Riccardo Conci and Peter G. Brodeur and Jason Gusdorf and Sourik Beltrán and Bita Behrouzi and Byron Crowe and Jacob Dockterman and Muzzammil Muhammad and Sarah Ohnigian and Andrew Sanchez and James A. Diao and Aashna P. Shah and Daniel Restrepo and Eric S. Rosenberg and Andrew S. Lea and Emily Glanton and Kimberly LeBlanc and Undiagnosed Diseases Network and Marinka Zitnik and Scott H. Podolsky and Zahir Kanjee and Raja-Elie E. Abdulnour and Jacob M. Koshy and Adam Rodman and Arjun K. Manrai},
      year={2026},
      eprint={2509.12194},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2509.12194},
}
```
