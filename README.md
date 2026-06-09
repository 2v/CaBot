# CaBot

CaBot reads a clinical case presentation and produces (1) a written **differential diagnosis** in
the style of the NEJM Clinicopathologic Conference and (2) an optional **video slideshow
presentation** that teaches the clinical reasoning.

This repository is the clean, public release used in the paper. A single `--version` flag selects
the exact model generation, and `--mode` selects what to generate.

## Versions

| Version | Line | Base model | Grounding | Presentation | Notes |
|---------|------|-----------|-----------|--------------|-------|
| `v1`    | main | o3        | literature + exemplar CPC retrieval | standard | Model used in the physician A/B test |
| `v1.1`  | main (**default**) | gpt-5.4 | literature + exemplar CPC retrieval | acknowledges missing information | Newest main-line model |
| `vr1`   | rare | gpt-5.4   | literature only (no exemplars) | acknowledges missing information | Tuned for UDN rare-disease application letters; runs without local data |
| `vs1`   | simple | o3      | literature only (no exemplars) | n/a (text only) | Simple QA / literature-search mode used for the NEJMBench QA & VQA benchmarks |
| `vs1.1` | simple | gpt-5.4 | literature only (no exemplars) | n/a (text only) | Same as `vs1`, newer base model |

The **simple line** (`vs1`, `vs1.1`) is not a differential generator: it answers a medical question
grounded only in `literature_search` results (markdown footnote citations), with no CPC formatting and
no exemplar retrieval — reproducing the configuration used for the QA & VQA benchmarks. It is text-only
(`--mode video`/`both` are rejected) and needs no local data. The case file you pass holds the question.

Each version is pinned to the source commit it was derived from (see `cabot_public_lib/versions.py`),
so its behavior can be cross-checked. All prompts are written out in full in
`cabot_public_lib/cabot_prompts.py` and are meant to be edited directly.

## Run modes (`--mode`)

- `text`  — differential diagnosis only (no system video dependencies required)
- `video` — slideshow only, generated from the case text (no differential attached)
- `both`  — differential, then a slideshow built from it (**default**)

## Setup

1. **Python deps**

   ```bash
   pip install -r requirements.txt
   ```

2. **API keys** — copy the template and fill it in:

   ```bash
   cp config.example.ini config.ini
   # set OPENAI_API_KEY and JWT_CLINICTRON in [main]
   ```

3. **System deps (only for `--mode video`/`both`)**: `pdflatex` (TeX Live/MacTeX with `beamer`),
   `pdftoppm` (poppler), and `ffmpeg`.

4. **Data (only for `v1`/`v1.1`)**: download the exemplar index
   `cpc_presentation_index_100.parquet` from the project website (dataset download page, "CaBot
   Exemplar Index" row) into `data/` (see `data/README.md`). `vr1` and `vs1`/`vs1.1` need no local
   data. **Note:** the public exemplar index covers only the **100 public CPCs**, so retrieved
   exemplars differ from the paper's full-corpus runs (see `data/README.md`).

## Usage

```bash
# Newest model, differential only — the quickest thing to try (no video deps, no local data needed
# if you use vr1; v1.1 needs the exemplar data)
python run_cabot.py --case examples/example_case.txt --output out/ --version vr1 --mode text

# Newest model, full pipeline (differential + slideshow)
python run_cabot.py --case examples/example_case.txt --output out/

# Original A/B-test model, text only
python run_cabot.py --case examples/example_case.txt --output out/ --version v1 --mode text

# Simple QA / literature-search mode (the case file holds the question; o3, as benchmarked)
python run_cabot.py --case examples/example_question.txt --output out/ --version vs1

# Reproduce the Nov 2025 Brigham video-only demo
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
--exclude-id        Case ID to exclude from exemplar retrieval + literature citations
--exclude-title     Case title to exclude (resolved to a case ID against the case database)
--config            Path to config.ini (default: config.ini)
--cpc-index         Parquet index for exemplar retrieval (default: data/cpc_presentation_index_100.parquet)
--nejm-cpcs-path    Dir with case images for video generation (default: data/nejm_cpcs)
--debug             Verbose model I/O
```

## Layout

```
cabot_public_lib/
├── cabot_prompts.py          # every version's prompts, in full (edit these)
├── versions.py               # version registry + linking source commits
├── cabot.py                  # text differential engine
├── video.py                  # slideshow pipeline (LaTeX -> PDF -> images -> TTS -> mp4)
└── cpc_presentation_store.py # exemplar CPC retrieval
run_cabot.py                  # CLI
```
