"""
CaBot text differential-diagnosis engine (public release).

Adapted from NEJMBench `cabot/cabot2.py@f5a08a76` — the iteration loop, reasoning capture,
literature-search tool use, exemplar retrieval, and progress-queue hooks are kept structurally
identical. The version-specific knobs that used to be hard-coded (prompt text, literature
result cap, reasoning effort, base model, affiliation, exemplar retrieval on/off) are now read
from a `VersionConfig` (see versions.py).
"""
import os
import json
import base64

from PIL import Image

from .openai_retry import call_with_retry
from .versions import VersionConfig

# Downloadable public exemplar index (100 public CPCs: presentation embeddings + differentials).
DEFAULT_CPC_INDEX = "data/cpc_presentation_index_100.parquet"
DEFAULT_NEJM_CPCS_PATH = "data/nejm_cpcs"


def encode_image(image_path):
    if image_path.lower().endswith(".tif"):
        img = Image.open(image_path)
        jpg_path = image_path.rsplit(".", 1)[0] + ".jpg"
        img.convert("RGB").save(jpg_path, "JPEG")
        image_path = jpg_path
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_tools_definition():
    """literature_search tool — verbatim from cabot2.py@f5a08a76."""
    return [
        {
            "type": "function",
            "name": "literature_search",
            "description": "Search peer-reviewed medical literature for evidence-based information. Include abstracts from a selection of 204 high-impact journals in clinical medicine.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query for medical literature"},
                    "min_citations": {"type": "integer", "description": "Minimum number of citations required for papers (default: 100). Higher values return more influential papers.", "default": 100},
                    "year_from": {"type": "integer", "description": "Earliest publication year to include in search (default: uses the year_min from CaBot initialization). Use recent years for current guidelines.", "minimum": 1800, "maximum": 2025},
                    "year_to": {"type": "integer", "description": "Latest publication year to include in search (default: None, no upper limit). Use to limit search to specific time periods.", "minimum": 1800, "maximum": 2025},
                },
                "required": ["query", "min_citations", "year_from", "year_to"],
                "additionalProperties": False,
            },
            "strict": True,
        }
    ]


def _reasoning_kwargs(base_model, effort):
    """Reasoning effort rule from cabot2.py: applied to o3 / gpt-5* reasoning models."""
    if effort is None:
        return {}
    if base_model == "o3" or base_model.startswith("gpt-5") or base_model.startswith("o"):
        return {"reasoning": {"effort": effort, "summary": "detailed"}}
    return {}


class CaBot:
    def __init__(self, client, version_config: VersionConfig, literature_store, cpc_store=None,
                 nejm_cpcs_path=DEFAULT_NEJM_CPCS_PATH):
        self.client = client
        self.cfg = version_config
        self.year_min = version_config.year_min
        self.year_max = version_config.year_max
        self.lit_top_k = version_config.lit_top_k
        self.lit_abstract_only = version_config.lit_abstract_only
        self.nejm_cpcs_path = nejm_cpcs_path

        # Both retrieval stores are built by the caller (see run_cabot.build_stores) and
        # injected fully loaded, so they load the same way and at the same point:
        #   - literature_store: required — every version uses the literature_search tool.
        #   - cpc_store: exemplar CPC retrieval, set only for versions that use it (v1, v1.1)
        #     and None otherwise (vr1, vs*). It also supplies the exemplar titles / DDx text.
        #     NOTE: it searches only the 100 public CPCs — unlike the original full-corpus retrieval.
        if literature_store is None:
            raise ValueError("literature_store is required: every CaBot version uses the "
                             "literature_search tool. Build a LiteratureSearchStore and pass it in.")
        self.literature_store = literature_store
        self.cpc_store = cpc_store

    @property
    def cpcs_data(self):
        """Raw exemplar records (titles / DDx / ids), or [] when exemplar retrieval is off."""
        return self.cpc_store.records if self.cpc_store else []

    # ------------------------------------------------------------------ literature
    def literature_search(self, query, min_citations=100, year_from=None, year_to=None,
                          exclude_id=None, abstract_only=True):
        if year_from is None:
            year_from = self.year_min
        if year_to is None:
            year_to = self.year_max

        try:
            result = self.literature_store.search(
                query, year_from=year_from, year_to=year_to,
                citations_min=min_citations, need_abstract=abstract_only)

            if result and "results" in result:
                filtered_results = []
                for paper in result["results"]:
                    doi = paper.get("doi", "") or ""
                    if exclude_id and (str(exclude_id).lower().strip() in doi.lower().strip()):
                        continue
                    filtered_results.append(paper)

                formatted_results = []
                for paper in filtered_results[: self.lit_top_k]:
                    formatted_results.append({
                        "title": paper.get("title", ""),
                        "authors": paper.get("authors", ""),
                        "journal": paper.get("journal", ""),
                        "year": paper.get("year", ""),
                        "abstract": paper.get("abstract", ""),
                        "citations": paper.get("citationCount", 0),
                    })
                return {"query": query, "results": formatted_results, "total_results": len(result.get("results", []))}
            return {"query": query, "results": [], "total_results": 0, "message": "No results found"}
        except Exception as e:
            return {"query": query, "error": f"Unexpected error: {str(e)}", "results": [], "total_results": 0}

    def get_similar_cases_with_metadata(self, case_text, top_k=3, exclude_id=None,
                                        year_min_override=None, year_max_override=None):
        if self.cpc_store is None:
            return "No similar cases provided for this analysis.", []
        try:
            search_year_min = year_min_override if year_min_override is not None else self.year_min
            search_year_max = year_max_override if year_max_override is not None else self.year_max
            scores, ids, texts, metadata = self.cpc_store.search_with_filters(
                query=case_text, year_min=search_year_min, year_max=search_year_max,
                k=top_k, exclude_id=exclude_id)

            formatted_cases, cases_metadata = [], []
            for score, case_id, _text, _meta in zip(scores, ids, texts, metadata):
                title = self.cpc_store.titles_by_id.get(case_id, f"Case {case_id}")
                ddx = self.cpc_store.ddx_by_id.get(case_id, "Differential diagnosis not available")
                if ddx and ddx != "Differential diagnosis not available":
                    formatted_cases.append(
                        f"**{title}**\n\n## Presentation of Case\n[Presentation of Case omitted for brevity]\n\n{ddx}\n\n")
                    cases_metadata.append({
                        "case_name": title, "case_id": case_id,
                        "similarity": float(score) if hasattr(score, "item") else score,
                    })
            return ("\n".join(formatted_cases) if formatted_cases else "No similar cases found."), cases_metadata
        except Exception as e:
            print(f"Error retrieving similar cases: {e}")
            return "Error retrieving similar cases.", []

    def invoke_functions_from_response(self, response, exclude_id=None, abstract_only=True):
        function_outputs = []
        function_calls = [rx for rx in response.output if rx.type == "function_call"]
        for function_call in function_calls:
            if function_call.name == "literature_search":
                args = json.loads(function_call.arguments)
                query = args["query"]
                min_citations = args.get("min_citations", 100)
                year_from = args.get("year_from", None)
                year_to = args.get("year_to", None)
                search_result = self.literature_search(query, min_citations, year_from, year_to, exclude_id, abstract_only)
                if search_result.get("error"):
                    tool_result = f"Error: {search_result['error']}"
                elif search_result.get("results"):
                    formatted_papers = []
                    for paper in search_result["results"]:
                        paper_text = (f"Title: {paper['title']}\nAuthors: {paper['authors']}\n"
                                      f"Journal: {paper['journal']} ({paper['year']})\nCitations: {paper['citations']}\n")
                        if paper["abstract"]:
                            paper_text += f"Abstract: {paper['abstract']}\n"
                        formatted_papers.append(paper_text)
                    year_to_display = f", year_to: {year_to}" if year_to is not None else ""
                    search_params = f"(min_citations: {min_citations}, year_from: {year_from or self.year_min}{year_to_display})"
                    tool_result = f"Found {len(search_result['results'])} papers for query '{query}' {search_params}:\n\n" + "\n---\n".join(formatted_papers)
                else:
                    tool_result = f"No papers found for query '{query}'"
            else:
                tool_result = f"Error: Unknown function {function_call.name}"
            function_outputs.append({"type": "function_call_output", "call_id": function_call.call_id, "output": tool_result})
        return function_outputs

    # ------------------------------------------------------------------ exclusion
    def resolve_exclusion(self, exclude_id=None, exclude_title=None):
        """Resolve a case to exclude from exemplar retrieval and literature citations.

        Accepts a case ID, a DOI, or a case title. Titles are resolved to a case ID against the
        local case database (search is performed against ID). Prints whether the case was found
        and will be excluded. Returns the exclude key (case ID / DOI substring) to apply, or None.
        """
        resolved = (exclude_id or "").strip() or None

        # title -> id (needs the local case database, i.e. a version that loads exemplars)
        if exclude_title:
            title = exclude_title.strip()
            if not self.cpcs_data:
                print(f"Cannot resolve title '{title}': no case database loaded for this "
                      f"version (exemplar retrieval is off).")
            else:
                exact = [c for c in self.cpcs_data if c.get("title", "").strip().lower() == title.lower()]
                subs = [c for c in self.cpcs_data if title.lower() in c.get("title", "").strip().lower()]
                hits = exact or subs
                if hits:
                    resolved = str(hits[0].get("id"))
                    print(f"Resolved title '{title}' -> case id '{resolved}' ({hits[0].get('title')})")
                    if len(hits) > 1:
                        print(f"   (note: {len(hits)} cases matched the title; using the first)")
                else:
                    print(f"No case matching title '{title}' found in the case database.")

        # confirm the exclusion against the local database (search against ID, then DOI)
        if resolved and self.cpcs_data:
            key = resolved.lower()
            hit = next((c for c in self.cpcs_data if str(c.get("id", "")).strip().lower() == key), None)
            if hit is None:
                hit = next((c for c in self.cpcs_data
                            if key in str(c.get("id", "")).strip().lower()
                            or key in str(c.get("doi", "")).strip().lower()), None)
            if hit:
                print(f"Exclusion confirmed: case '{hit.get('id')}' ({hit.get('title', '')}) will be "
                      f"excluded from exemplar case retrieval and from literature citations.")
            else:
                print(f"'{resolved}' was not found in the local case database; it will still be "
                      f"applied as an ID/DOI filter to literature search and vector retrieval.")
        elif resolved:
            print(f"Exclusion '{resolved}' will be applied as a DOI/ID filter to literature search "
                  f"(no local case database loaded to confirm against).")
        return resolved

    # ------------------------------------------------------------------ run
    def run(self, presentation_of_case, images=None, debug=True, max_iterations=None,
            exclude_id=None, exclude_title=None, year_anchor=None, base_model=None,
            progress_queue=None, cancellation_event=None):
        cfg = self.cfg
        images = images or []
        max_iterations = max_iterations if max_iterations is not None else cfg.max_iterations
        year_anchor = year_anchor if year_anchor is not None else cfg.year_anchor
        base_model = base_model or cfg.base_model
        agent = cfg.agent

        # Resolve & confirm the case to exclude (by ID, DOI, or title) before retrieval.
        exclude_id = self.resolve_exclusion(exclude_id=exclude_id, exclude_title=exclude_title)

        model_kwargs = _reasoning_kwargs(base_model, cfg.reasoning_effort)

        # exemplar retrieval (only when enabled for this version)
        if cfg.use_similar_cases:
            year_min_search = year_anchor - 2 if year_anchor is not None else None
            year_max_search = year_anchor + 2 if year_anchor is not None else None
            similar_cases_text, similar_cases_metadata = self.get_similar_cases_with_metadata(
                case_text=presentation_of_case, top_k=2, exclude_id=exclude_id,
                year_min_override=year_min_search, year_max_override=year_max_search)
        else:
            similar_cases_text, similar_cases_metadata = "No similar cases provided for this analysis.", []

        # apply UDN-style input wrapper if this version uses one
        case_text = presentation_of_case
        if cfg.input_wrapper:
            case_text = cfg.input_wrapper.replace("{application}", presentation_of_case)

        # build the question by literal token substitution (robust to stray braces in prompts)
        question = cfg.ddx_prompt
        for token, value in (("{year_max}", str(self.year_max)),
                             ("{similar_cases}", similar_cases_text),
                             ("{case}", case_text),
                             ("{max_iterations}", str(max_iterations))):
            question = question.replace(token, value)

        conversation = [{"role": "user", "type": "message", "content": question}]

        for image in images:
            try:
                image_path = os.path.join(self.nejm_cpcs_path, image["path"])
                base64_image = encode_image(image_path)
                if image["type"] == "table":
                    caption_text = f"Here is Table {int(image['rid'].lstrip('t'))}"
                elif image["type"] == "fig":
                    caption_text = f"Here is Figure {int(image['rid'].lstrip('f'))}"
                else:
                    raise Exception("Unknown image type passed")
                conversation.append({"role": "user", "content": [
                    {"type": "input_text", "text": caption_text},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"},
                ]})
            except (FileNotFoundError, OSError, IOError) as e:
                if debug:
                    print(f"Warning: Could not load image '{image.get('path')}': {e}. Skipping.")
                continue

        tools = get_tools_definition() if agent else None
        total_tokens_used = 0
        iterations = 0

        if debug:
            print(f"{'*' * 79}\nUser message: {question}\n{'*' * 79}")

        if progress_queue:
            progress_queue.put({"type": "progress", "iteration": 0, "max_iterations": max_iterations,
                                "status": "Starting analysis...", "stage": "initialization"})
            if similar_cases_metadata:
                progress_queue.put({"type": "similar_cases", "similar_cases": similar_cases_metadata})

        while iterations < max_iterations:
            iterations += 1
            if cancellation_event and cancellation_event.is_set():
                return {"output": "Analysis cancelled by user.", "similar_cases": similar_cases_metadata}
            if progress_queue:
                progress_queue.put({"type": "progress", "iteration": iterations, "max_iterations": max_iterations,
                                    "status": f"Processing iteration {iterations}...", "stage": "reasoning"})
            if debug:
                print(f"\n{'='*80}\nITERATION {iterations}\n{'='*80}")

            if agent:
                response = call_with_retry(self.client.responses.create,
                                           model=base_model, input=conversation, tools=tools, **model_kwargs)
            else:
                response = call_with_retry(self.client.responses.create,
                                           model=base_model, input=conversation, **model_kwargs)

            total_tokens_used += response.usage.total_tokens

            if not agent:
                if debug:
                    print(f"\n[NON-AGENT RESPONSE]:\n{response.output_text}")
                if progress_queue:
                    progress_queue.put({"type": "progress", "iteration": iterations, "max_iterations": max_iterations,
                                        "status": "Analysis complete!", "stage": "complete"})
                return {"output": response.output_text, "similar_cases": similar_cases_metadata}

            reasoning = [rx.to_dict() for rx in response.output if rx.type == "reasoning"]
            function_calls = [rx.to_dict() for rx in response.output if rx.type == "function_call"]
            messages = [rx.to_dict() for rx in response.output if rx.type == "message"]

            if len(reasoning) > 0:
                if debug and "summary" in reasoning[0] and len(reasoning[0]["summary"]) > 0:
                    print(reasoning[0]["summary"][0]["text"])
                conversation.extend(reasoning)

            if len(function_calls) > 0:
                if debug:
                    for fc in function_calls:
                        print(f"\n[TOOL CALL]: {fc['name']} with args: {fc['arguments']}")
                if progress_queue:
                    progress_queue.put({"type": "progress", "iteration": iterations, "max_iterations": max_iterations,
                                        "status": f"Searching literature ({len(function_calls)} searches)...", "stage": "literature_search"})
                function_outputs = self.invoke_functions_from_response(
                    response, exclude_id=exclude_id, abstract_only=self.lit_abstract_only)
                if debug:
                    for fo in function_outputs:
                        print(f"\n[TOOL RESULT]: {fo['output']}")
                interleaved = [val for pair in zip(function_calls, function_outputs) for val in pair]
                conversation.extend(interleaved)

            if len(messages) > 0:
                if debug:
                    print(f"\n[ASSISTANT RESPONSE]:\n{response.output_text}")
                conversation.extend(messages)
                if "[Differential Diagnosis]" in response.output_text:
                    if debug:
                        print("\n[FINAL ANSWER DETECTED]")
                    if progress_queue:
                        progress_queue.put({"type": "progress", "iteration": iterations, "max_iterations": max_iterations,
                                            "status": "Analysis complete!", "stage": "complete"})
                    return {"output": response.output_text, "similar_cases": similar_cases_metadata}

            if len(function_calls) == 0:
                if "[Differential Diagnosis]" not in response.output_text:
                    if debug:
                        print("\n[RESPONSE WITHOUT TAG -> REQUESTING REFORMAT]")
                    conversation.append({"role": "user", "type": "message", "content":
                        "Your response does not include the required [Differential Diagnosis] tag. Please continue your research if needed, or reformat your final answer to include the [Differential Diagnosis] tag followed by your complete differential diagnosis in markdown format."})
                else:
                    break

        if progress_queue:
            progress_queue.put({"type": "progress", "iteration": iterations, "max_iterations": max_iterations,
                                "status": "Maximum iterations reached", "stage": "error"})
            progress_queue.put(None)
        return {"output": "Maximum iterations reached without a final answer.", "similar_cases": similar_cases_metadata}

    # ------------------------------------------------------------- simple QA / literature
    def run_simple_literature(self, question=None, messages=None, debug=True, base_model=None,
                              max_iterations=None, abstract_only=None):
        """Simple QA / literature-search mode (mode="simple_qa"; vs1, vs1.1).

        Answers a medical question grounded only in literature_search results — no CPC
        formatting and no exemplar retrieval. Faithful to cabot2_old.py@f404f0dd
        run_simple_literature as driven by the NEJMBench QA & VQA benchmarks.
        """
        cfg = self.cfg
        if abstract_only is None:
            abstract_only = self.lit_abstract_only
        base_model = base_model or cfg.base_model
        max_iterations = max_iterations if max_iterations is not None else cfg.max_iterations
        model_kwargs = _reasoning_kwargs(base_model, cfg.reasoning_effort)

        # {max_tool_calls} is the tool-call budget the model is told about (= iterations - 1).
        max_tool_calls = max_iterations - 1
        system_prompt = cfg.simple_qa_system_prompt or ""
        system_prompt = system_prompt.replace("{max_tool_calls}", str(max_tool_calls)) \
                                     .replace("{max_iterations}", str(max_iterations))

        if messages is None:
            if question is None:
                raise ValueError("run_simple_literature requires either question or messages")
            messages = [{"role": "user", "type": "message", "content": question}]

        conversation = [{"role": "system", "type": "message", "content": system_prompt}]
        conversation.extend(messages)

        tools = get_tools_definition()
        total_tokens_used = 0
        iterations = 0

        if debug:
            print(f"Running simple literature search mode (model={base_model}, "
                  f"max_iterations={max_iterations})...")

        while iterations < max_iterations:
            iterations += 1
            if debug:
                print(f"\n{'='*80}\nITERATION {iterations}\n{'='*80}")

            response = call_with_retry(self.client.responses.create,
                                       model=base_model, input=conversation, tools=tools, **model_kwargs)
            total_tokens_used += response.usage.total_tokens

            reasoning = [rx.to_dict() for rx in response.output if rx.type == "reasoning"]
            function_calls = [rx.to_dict() for rx in response.output if rx.type == "function_call"]
            messages_out = [rx.to_dict() for rx in response.output if rx.type == "message"]

            if len(reasoning) > 0:
                conversation.extend(reasoning)

            if len(function_calls) > 0:
                if debug:
                    for fc in function_calls:
                        print(f"\n[TOOL CALL]: {fc['name']} with args: {fc['arguments']}")
                function_outputs = self.invoke_functions_from_response(response, exclude_id=None, abstract_only=abstract_only)
                # tell the model how many tool calls remain (verbatim from cabot2_old.py)
                remaining_iterations = max_iterations - iterations - 1
                for fo in function_outputs:
                    fo['output'] += (f"\n\n---\nIMPORTANT: You have {remaining_iterations} tool calls "
                                     f"remaining (including this one). Please plan accordingly and "
                                     f"provide your final answer before exhausting all calls.")
                if debug:
                    for fo in function_outputs:
                        print(f"\n[TOOL RESULT]: {fo['output']}")
                interleaved = [val for pair in zip(function_calls, function_outputs) for val in pair]
                conversation.extend(interleaved)

            if len(messages_out) > 0:
                # simple mode returns on any assistant message (no [Differential Diagnosis] tag)
                if debug:
                    print(f"\n[ASSISTANT RESPONSE]:\n{response.output_text}")
                conversation.extend(messages_out)
                return {"output": response.output_text, "model_used": base_model,
                        "total_tokens": total_tokens_used, "iterations": iterations}

            if len(function_calls) == 0:  # no more tool calls and no message -> done
                break

        return {"output": "Maximum iterations reached without a final answer.",
                "model_used": base_model, "total_tokens": total_tokens_used, "iterations": iterations}
