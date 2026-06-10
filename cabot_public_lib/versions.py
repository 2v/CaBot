"""
CaBot-Public version registry.

Three independent version lines:
  - main line (NEJM CPC-style cases):   v1  ->  v1.1  (default)
  - rare line (UDN rare-disease cases): vr1
  - simple line (literature-grounded QA): vs1 -> vs1.1

The simple line reproduces the "simple QA / literature search" configuration used for the
NEJMBench QA & VQA benchmarks: no CPC formatting, no exemplar retrieval — just answer a question,
grounded in the literature_search tool. These versions set mode="simple_qa".

Literature-search scope differs by line (``lit_abstract_only``):
  - DDx line (v1, v1.1, vr1): search ONLY abstract-bearing works (~1.5M of the 3.47M index)
    and return the top-5 papers with abstracts.
  - simple line (vs1, vs1.1): search the full corpus with dual retrieval — top-5 title-only
    results followed by top-5 with abstracts.

Each VersionConfig records the NEJMBench `repo_commit` that pins the exact behavior, so any
version can be cross-checked against the source it was derived from. Run mode (text / video /
both) is an orthogonal CLI flag, not a version property (the simple line is text-only).
"""
from dataclasses import dataclass
from typing import Optional

from . import cabot_prompts as P


@dataclass(frozen=True)
class VersionConfig:
    # --- identity (required) ---
    name: str
    line: str                       # "main" | "rare" | "simple"
    description: str
    repo_commit: str                # linking NEJMBench commit (traceability)

    # --- model / inference, shared by all modes (required) ---
    base_model: str                 # default base model (CLI can override)
    reasoning_effort: Optional[str] # "low" | "high" | None  (o3 -> low, gpt-5* -> high)
    agent: bool                     # literature_search tool enabled
    lit_top_k: int                  # literature results kept per search
    lit_abstract_only: bool         # literature_search restricted to abstract-bearing works
    year_min: int
    year_max: int
    max_iterations: int

    # --- mode selector ---
    mode: str = "ddx"               # "ddx" (CPC/UDN differential) | "simple_qa" (literature QA)

    # --- text DDx (mode="ddx") ---
    use_similar_cases: bool = False # exemplar CPC retrieval (needs vector store)
    ddx_prompt: Optional[str] = None        # full prompt template (from cabot_prompts)
    input_wrapper: Optional[str] = None     # wraps the case text before DDx (UDN); else None
    year_anchor: Optional[int] = None       # exemplar/temporal anchor used at original run time

    # --- simple QA / literature search (mode="simple_qa") ---
    simple_qa_system_prompt: Optional[str] = None  # system prompt; {max_tool_calls} filled at run time

    # --- presentation / video (mode="ddx") ---
    presentation_style: Optional[str] = None       # "monolithic" (v1) | "split" (v1.1, vr1)
    presentation_prompt: Optional[str] = None       # monolithic style: single prompt
    presentation_with_ddx_prefix: Optional[str] = None
    presentation_without_ddx_prefix: Optional[str] = None
    presentation_body: Optional[str] = None
    default_video_base_model: Optional[str] = None  # model used for the slideshow generation

    # default run mode when the canonical experiment skipped video
    canonical_mode: str = "both"    # informational; CLI default mode is separate


VERSIONS = {
    # ---- main line ----
    "v1": VersionConfig(
        name="v1",
        line="main",
        description="Original CaBot used in the physician A/B test (NEJM CPC cases). o3, "
                    "literature-grounded, exemplar-retrieval, standard presentation.",
        repo_commit="f5a08a76",
        base_model="o3",
        reasoning_effort="low",
        agent=True,
        use_similar_cases=True,
        lit_top_k=5,
        lit_abstract_only=True,   # DDx line: search only abstract-bearing works (~1.5M)
        ddx_prompt=P.V1_DDX_PROMPT,
        input_wrapper=None,
        year_min=2000,
        year_max=2025,
        year_anchor=2022,
        max_iterations=25,
        presentation_style="monolithic",
        presentation_prompt=P.V1_PRESENTATION_PROMPT,
        presentation_with_ddx_prefix=None,
        presentation_without_ddx_prefix=None,
        presentation_body=None,
        default_video_base_model="o3",
    ),
    "v1.1": VersionConfig(
        name="v1.1",
        line="main",
        description="Newest main-line CaBot: missing information acknowledged up front in BOTH "
                    "the text differential and the slideshow. gpt-5.4, literature-grounded, "
                    "exemplar-retrieval, missing-info presentation (Nov 2025 Brigham).",
        repo_commit="c09c6952",  # presentation; text-DDx prompt is authored (see cabot_prompts)
        base_model="gpt-5.4",
        reasoning_effort="high",
        agent=True,
        use_similar_cases=True,
        lit_top_k=5,
        lit_abstract_only=True,   # DDx line: search only abstract-bearing works (~1.5M)
        ddx_prompt=P.V1_1_DDX_PROMPT,
        input_wrapper=None,
        year_min=2000,
        year_max=2025,
        year_anchor=2022,
        max_iterations=30,
        presentation_style="split",
        presentation_prompt=None,
        presentation_with_ddx_prefix=P.PRESENTATION_WITH_DDX_PREFIX,
        presentation_without_ddx_prefix=P.PRESENTATION_WITHOUT_DDX_PREFIX,
        presentation_body=P.PRESENTATION_BODY_MISSING_INFO_NOV,
        default_video_base_model="gpt-5.4",
    ),
    # ---- rare line ----
    "vr1": VersionConfig(
        name="vr1",
        line="rare",
        description="CaBot-Rare: UDN rare-disease application letters. gpt-5.4, literature-"
                    "grounded, NO exemplar retrieval, UDN input wrapper, missing-info "
                    "presentation. Canonical run was text-only.",
        repo_commit="4fe60742",
        base_model="gpt-5.4",
        reasoning_effort="high",
        agent=True,
        use_similar_cases=False,
        lit_top_k=5,
        lit_abstract_only=True,   # DDx line: search only abstract-bearing works (~1.5M)
        ddx_prompt=P.VR1_DDX_PROMPT,
        input_wrapper=P.UDN_APPLICATION_WRAPPER,
        year_min=2000,
        year_max=2024,
        year_anchor=2024,
        max_iterations=30,
        presentation_style="split",
        presentation_prompt=None,
        presentation_with_ddx_prefix=P.PRESENTATION_WITH_DDX_PREFIX,
        presentation_without_ddx_prefix=P.PRESENTATION_WITHOUT_DDX_PREFIX,
        presentation_body=P.PRESENTATION_BODY_MISSING_INFO_UDN,
        default_video_base_model="gpt-5.4",
        canonical_mode="text",
    ),
    # ---- simple line (literature-grounded QA, as run for the QA & VQA benchmarks) ----
    "vs1": VersionConfig(
        name="vs1",
        line="simple",
        description="Simple QA / literature-search CaBot as run for the NEJMBench QA & VQA "
                    "benchmarks: answer a medical question grounded ONLY in literature_search "
                    "results, with markdown footnote citations. No CPC formatting, no exemplar "
                    "retrieval. o3, matching the original benchmark.",
        repo_commit="f404f0dd",
        base_model="o3",
        reasoning_effort="high",        # run_simple_literature used effort=high for o3/gpt-5
        agent=True,
        lit_top_k=5,
        lit_abstract_only=False,  # simple line: dual retrieval (title-only + abstract results)
        year_min=2000,
        year_max=2024,
        max_iterations=5,               # benchmark used 5 iterations (max_tool_calls = 4)
        mode="simple_qa",
        simple_qa_system_prompt=P.SIMPLE_QA_SYSTEM_PROMPT,
        canonical_mode="text",
    ),
    "vs1.1": VersionConfig(
        name="vs1.1",
        line="simple",
        description="Simple QA / literature-search CaBot, gpt-5.4 variant of vs1 (same "
                    "configuration as the QA & VQA benchmark mode, newer base model).",
        repo_commit="f404f0dd",
        base_model="gpt-5.4",
        reasoning_effort="high",
        agent=True,
        lit_top_k=5,
        lit_abstract_only=False,  # simple line: dual retrieval (title-only + abstract results)
        year_min=2000,
        year_max=2025,
        max_iterations=5,
        mode="simple_qa",
        simple_qa_system_prompt=P.SIMPLE_QA_SYSTEM_PROMPT,
        canonical_mode="text",
    ),
}

DEFAULT_VERSION = "v1.1"


def get_version(name: str) -> VersionConfig:
    if name not in VERSIONS:
        raise ValueError(
            f"Unknown version '{name}'. Available: {', '.join(VERSIONS)}"
        )
    return VERSIONS[name]
