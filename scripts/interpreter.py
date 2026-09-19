"""
interpreter.py — Main QueryInterpreter orchestrator for SatQuery AI.

Coordinates LLM prompt execution, defensive JSON parsing & normalization,
schema validation, fallback dispatch, and error diagnostics. Designed for
standalone operation (llm_fn=None) and plug-and-play integration with
Person B's local Qwen3.5-2B model wrapper.
"""

import json
import logging
import re
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from .keyword_fallback import classify_query_fallback, extract_task_sequence, DOMAIN_KEYWORDS_3TIER
    from .prompt_template import build_prompt
    from .schema import Domain, InterpretedQuery, Sensor, Source, Task
    from .clarification_memory import ClarificationMemory
except ImportError:
    from keyword_fallback import classify_query_fallback, extract_task_sequence, DOMAIN_KEYWORDS_3TIER
    from prompt_template import build_prompt
    from schema import Domain, InterpretedQuery, Sensor, Source, Task
    from clarification_memory import ClarificationMemory

# Below this confidence, the interpreter still returns its best guess (never
# blocks the pipeline) but flags needs_clarification=True so the caller can
# decide to ask the user instead of silently trusting a shaky guess.
CLARIFICATION_CONFIDENCE_THRESHOLD = 0.5

logger = logging.getLogger("SatQuery.Interpreter")


# -----------------------------------------------------------------------
# Keyword-engine arbitration helpers
#
# keyword_fallback.py contains a hand-tuned, high-precision keyword engine
# (tiered domain scoring with hand-picked domain-defining phrases; strict
# regex triggers for grounding/change_vqa/fusion). It was originally used
# only as an emergency fallback when the LLM fails entirely. Benchmarking
# showed the LLM path frequently mis-classifies domain and adds tasks
# (grounding, change_vqa) that the query text doesn't actually signal --
# exactly the failure modes this keyword engine was hand-built to avoid.
# These two functions use it as a cross-check/correction layer on TOP of
# successful LLM output, not just as a fallback for failed output.
# -----------------------------------------------------------------------

def _tier3_domain_override(raw_query: str, llm_domain: str) -> Optional[str]:
    """
    Checks only the highest-precision (tier 3) domain keyword phrases --
    e.g. "thermal power station", "burn scar severity" -- which are
    hand-picked to be unambiguous. If a tier-3 phrase for a DIFFERENT
    domain than the LLM chose is present in the query, that's a strong
    enough signal to override the LLM's choice. Tier 1/2 (generic terms)
    are intentionally NOT used here -- those are weak signals and
    overriding on them risks being wrong more often than the LLM.
    Returns the override domain value, or None if no override applies.
    """
    q_lower = raw_query.lower()
    for domain, tiers in DOMAIN_KEYWORDS_3TIER.items():
        if domain.value == llm_domain:
            continue
        for kw in tiers.get(3, []):
            if re.search(r"\b" + re.escape(kw) + r"\b", q_lower):
                return domain.value
    return None


def _cross_check_task_sequence(raw_query: str, llm_tasks: List[str]) -> List[str]:
    """
    grounding, change_vqa, and fusion_analysis all have strict, hand-tuned
    regex triggers in extract_task_sequence() (e.g. change_vqa requires an
    explicit comparison word like "compare"/"since"/"before and after" --
    it is never added just because a domain is often time-sensitive).
    If the LLM included one of these three tasks but the query contains
    NONE of that task's trigger phrases, drop it -- this is exactly the
    "Qwen added change_vqa/grounding without a real signal" failure mode
    from benchmarking.

    Separately: captioning also has strong, well-tuned triggers ("describe",
    "what does X look like", "what is shown/visible", "summarize",
    "overview", "tell me what you see"). Benchmarking showed the opposite
    failure mode for this pair specifically -- Qwen frequently answers "vqa"
    for queries that are actually open-ended scene descriptions phrased as
    a question ("what is shown in this image?" is captioning, not a
    single-fact vqa lookup). If the LLM said "vqa" but the query strongly
    triggers captioning and captioning wasn't already in the list, swap it.

    vqa is otherwise left untouched when NEITHER of the above applies: the
    remaining vqa-vs-captioning boundary (closed factual questions vs. open
    description) is a semantic distinction keyword matching handles far
    less reliably than the LLM does, so we don't second-guess it further.
    """
    fallback_tasks = {t.value for t in extract_task_sequence(raw_query)}
    strict_tasks = {Task.GROUNDING.value, Task.CHANGE_VQA.value, Task.FUSION_ANALYSIS.value}

    checked = [
        t for t in llm_tasks
        if t not in strict_tasks or t in fallback_tasks
    ]

    # vqa -> captioning correction, only when captioning wasn't already
    # correctly identified and the query strongly signals it.
    if (
        Task.VQA.value in checked
        and Task.CAPTIONING.value not in checked
        and Task.CAPTIONING.value in fallback_tasks
    ):
        checked = [
            Task.CAPTIONING.value if t == Task.VQA.value else t
            for t in checked
        ]

    return checked or [Task.VQA.value]  # never return an empty list



class QueryInterpreter:
    """
    Interprets natural-language remote sensing queries into structured JSON
    adhering strictly to schema.InterpretedQuery.

    Attributes:
        llm_fn: Optional callable (prompt: str) -> raw_completion_str.
        query_bank: Optional query bank dictionary list for evaluation/reference.
        last_error: Diagnostic record of the most recent parsing/LLM failure.
    """

    def __init__(
        self,
        query_bank: Optional[List[Dict[str, Any]]] = None,
        llm_fn: Optional[Callable[[str], str]] = None,
        memory: Optional[ClarificationMemory] = None,
    ) -> None:
        self.query_bank = query_bank
        self.llm_fn = llm_fn
        self.last_error: Optional[Dict[str, Any]] = None
        # memory=None (default) still creates a real, persistent
        # ClarificationMemory backed by clarification_memory.json in this
        # folder -- pass an explicit ClarificationMemory(path=...) if you
        # want a different location, or one per-user/session.
        self.memory = memory if memory is not None else ClarificationMemory()

    def interpret(self, query: str) -> InterpretedQuery:
        """
        Interprets a user query into an InterpretedQuery object.

        Order of operations:
          0. Check clarification memory first. If this exact query (or a
             near-identical one -- see clarification_memory.py's matching
             notes) was clarified by a user before, return that instantly,
             with zero LLM calls.
          1. If llm_fn is provided:
             a. Builds the few-shot prompt.
             b. Invokes llm_fn(prompt).
             c. Defensively parses, repairs, and normalizes the JSON output.
             d. Validates against schema.InterpretedQuery.
             e. Returns Source.LLM (or Source.HYBRID if fields were backfilled).
          2. If llm_fn is None or if any step fails:
             - Records diagnostic error details in self.last_error.
             - Dispatches to keyword_fallback.classify_query_fallback().
             - Returns result with Source.KEYWORD_FALLBACK.
          3. Either way, if the result's confidence is below
             CLARIFICATION_CONFIDENCE_THRESHOLD, needs_clarification is set
             True. The pipeline is NEVER blocked -- callers decide whether
             to act on that flag (e.g. by asking the user a question and
             calling apply_user_clarification() with the answer).
        """
        clean_query = query.strip()

        # Step 0: memory check -- a previously clarified answer always wins,
        # skips everything else (LLM call included).
        remembered = self.memory.lookup(clean_query)
        if remembered is not None:
            return remembered

        # Path 1: Standalone / Fallback-only mode (No GPU/LLM required)
        if self.llm_fn is None:
            result = classify_query_fallback(clean_query)
            return self._flag_if_low_confidence(result)

        # Path 2: LLM Inference with defensive JSON recovery
        try:
            prompt = build_prompt(clean_query)
            raw_response = self.llm_fn(prompt)

            # Defensive parsing, normalization, and repair
            parsed_dict, backfilled_fields = self._parse_and_repair_llm_json(
                raw_text=raw_response,
                raw_query=clean_query
            )

            # Validate against Pydantic schema contract
            interpreted = InterpretedQuery.model_validate(parsed_dict)

            # Determine source: HYBRID if backfilled, otherwise LLM
            if backfilled_fields:
                interpreted.source = Source.HYBRID
            else:
                interpreted.source = Source.LLM

            # Ensure raw_query matches original query
            interpreted.raw_query = clean_query
            return self._flag_if_low_confidence(interpreted)

        except Exception as err:
            # Record complete failure context for Person B debugging
            self.last_error = {
                "raw_query": clean_query,
                "raw_llm_output": raw_response if "raw_response" in locals() else None,
                "error_type": type(err).__name__,
                "error_message": str(err),
                "traceback": traceback.format_exc(),
            }
            logger.warning(
                f"LLM interpretation failed for query '{clean_query}': {err}. "
                f"Falling back to keyword classifier."
            )

            # Safety net: Guarantee valid InterpretedQuery output via fallback
            fallback_res = classify_query_fallback(clean_query)
            fallback_res.source = Source.KEYWORD_FALLBACK
            return self._flag_if_low_confidence(fallback_res)

    def _flag_if_low_confidence(self, result: InterpretedQuery) -> InterpretedQuery:
        """Sets needs_clarification=True when confidence is below threshold.
        Never blocks -- result is still returned as the best available guess."""
        if result.confidence < CLARIFICATION_CONFIDENCE_THRESHOLD:
            result.needs_clarification = True
        return result

    def apply_user_clarification(
        self,
        query: str,
        corrected: InterpretedQuery,
    ) -> InterpretedQuery:
        """
        Call this once a person has answered a clarifying question for a
        low-confidence query. Marks the result as user-clarified, sets
        confidence to 1.0 (a human confirmed it), clears
        needs_clarification, and permanently saves it to clarification
        memory -- so this exact query (or a near-identical one, see
        clarification_memory.py) is answered instantly from then on,
        without ever hitting the LLM or asking the user again.

        Typical usage in a UI/orchestrator layer:
            result = interpreter.interpret(query)
            if result.needs_clarification:
                # show result's best guess to the user, let them correct
                # whichever fields were wrong, then:
                corrected = result.model_copy(update={"domain": user_answer})
                interpreter.apply_user_clarification(query, corrected)
        """
        corrected.source = Source.USER_CLARIFIED
        corrected.confidence = 1.0
        corrected.needs_clarification = False
        corrected.raw_query = query.strip()
        self.memory.remember(query, corrected)
        return corrected

    def get_last_error_details(self) -> Optional[Dict[str, Any]]:
        """
        Returns structured diagnostic details of the last failure encountered
        during LLM execution or JSON parsing. Useful for Person B to report
        and debug model misbehavior.
        """
        return self.last_error

    # -----------------------------------------------------------------------
    # Defensive JSON Parsing & Repair Engine
    # -----------------------------------------------------------------------

    def _parse_and_repair_llm_json(
        self,
        raw_text: str,
        raw_query: str
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Extracts, repairs, normalizes, and backfills JSON from raw LLM output.

        Handles common small-LLM quirks:
        - Markdown code fences (```json ... ```)
        - Conversational preambles ("Sure, here's the JSON:")
        - Trailing commas in arrays and objects
        - Single quotes instead of double quotes
        - Python literals (None, True, False) instead of JSON literals (null, true, false)
        - Field name aliases (tasks -> task_sequence, date -> date_or_time_reference)
        - Type discrepancies (string task instead of list of tasks)
        - Missing critical fields (backfilled from keyword fallback classifier)

        Returns:
            Tuple of (repaired_dict, list_of_backfilled_field_names)
        """
        text = raw_text.strip()

        # Step 1: Strip markdown code blocks if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

        # Step 2: Extract outermost JSON object if wrapped in conversational text
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if json_match:
            text = json_match.group(0)

        # Step 3: Normalize common JSON malformations
        # Replace Python None, True, False if unquoted
        text = re.sub(r"\bNone\b", "null", text)
        text = re.sub(r"\bTrue\b", "true", text)
        text = re.sub(r"\bFalse\b", "false", text)

        # Remove trailing commas in objects and arrays e.g. {"a": 1, } or ["vqa", ]
        text = re.sub(r",\s*([\]}])", r"\1", text)

        # Step 4: Attempt standard JSON parse
        parsed: Dict[str, Any]
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            # Attempt aggressive repair: replace single quotes with double quotes
            # while preserving apostrophes inside words if possible
            repaired = text.replace("'", '"')
            repaired = re.sub(r",\s*([\]}])", r"\1", repaired)
            parsed = json.loads(repaired)

        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")

        # Step 5: Normalize field names (handle minor key aliases)
        key_aliases = {
            "task": "task_sequence",
            "tasks": "task_sequence",
            "task_list": "task_sequence",
            "date": "date_or_time_reference",
            "time": "date_or_time_reference",
            "time_reference": "date_or_time_reference",
            "date_reference": "date_or_time_reference",
            "target": "target_object",
            "object": "target_object",
            "query": "raw_query",
        }
        for alias, target in key_aliases.items():
            if alias in parsed and target not in parsed:
                parsed[target] = parsed.pop(alias)

        # Step 6: Type normalization and value coercion
        # Ensure task_sequence is a list
        if "task_sequence" in parsed:
            if isinstance(parsed["task_sequence"], str):
                parsed["task_sequence"] = [parsed["task_sequence"]]
            elif isinstance(parsed["task_sequence"], list):
                # Clean up string items in task_sequence
                clean_tasks = []
                for t in parsed["task_sequence"]:
                    t_str = str(t).strip().lower().replace("-", "_").replace(" ", "_")
                    # Map common task variations
                    if "change" in t_str:
                        clean_tasks.append(Task.CHANGE_VQA.value)
                    elif "ground" in t_str or "locate" in t_str:
                        clean_tasks.append(Task.GROUNDING.value)
                    elif "caption" in t_str or "describe" in t_str or "summary" in t_str:
                        clean_tasks.append(Task.CAPTIONING.value)
                    elif "fuse" in t_str or "fusion" in t_str:
                        clean_tasks.append(Task.FUSION_ANALYSIS.value)
                    elif "vqa" in t_str or "question" in t_str:
                        clean_tasks.append(Task.VQA.value)
                    else:
                        clean_tasks.append(t_str)
                # Deduplicate while preserving order
                dedup_tasks = []
                for t in clean_tasks:
                    if t not in dedup_tasks:
                        dedup_tasks.append(t)
                parsed["task_sequence"] = dedup_tasks or [Task.VQA.value]
        else:
            parsed["task_sequence"] = [Task.VQA.value]

        # Step 6a2: Cross-check task_sequence against the keyword engine's
        # strict triggers for grounding/change_vqa/fusion_analysis. Drops
        # any of these three the LLM added without real textual signal.
        parsed["task_sequence"] = _cross_check_task_sequence(raw_query, parsed["task_sequence"])

        # Normalize domain
        if "domain" in parsed and isinstance(parsed["domain"], str):
            d_str = parsed["domain"].strip().lower()
            valid_domains = {d.value for d in Domain}
            parsed["domain"] = d_str if d_str in valid_domains else Domain.GENERAL.value
        else:
            parsed["domain"] = Domain.GENERAL.value

        # Step 6c: Tier-3 keyword override for domain. Only fires on
        # hand-picked, unambiguous domain-defining phrases -- see
        # _tier3_domain_override() docstring.
        domain_override = _tier3_domain_override(raw_query, parsed["domain"])
        if domain_override is not None:
            parsed["domain"] = domain_override

        # Normalize sensor
        if "sensor" in parsed and isinstance(parsed["sensor"], str):
            s_str = parsed["sensor"].strip().lower()
            valid_sensors = {s.value for s in Sensor}
            if s_str in ("radar", "sar_sensor"):
                s_str = Sensor.SAR.value
            elif s_str in ("rgb", "visual", "multispectral"):
                s_str = Sensor.OPTICAL.value
            elif s_str in ("optical_and_sar", "fusion"):
                s_str = Sensor.BOTH.value
            parsed["sensor"] = s_str if s_str in valid_sensors else Sensor.UNSPECIFIED.value
        else:
            parsed["sensor"] = Sensor.UNSPECIFIED.value

        # Normalize confidence
        if "confidence" in parsed:
            try:
                parsed["confidence"] = max(0.0, min(1.0, float(parsed["confidence"])))
            except (ValueError, TypeError):
                parsed["confidence"] = 0.85
        else:
            parsed["confidence"] = 0.85

        # Step 6b: Enforce target_object contract — schema.py is explicit that
        # target_object must be null for non-grounding tasks (see the field's
        # description). keyword_fallback.extract_target_object() already
        # enforces this correctly; the LLM path did not, which was the
        # single largest source of field mismatches in benchmarking (Qwen
        # frequently names the subject of a VQA/captioning query even though
        # it isn't a grounding request). Force the contract here regardless
        # of what the LLM emitted.
        if Task.GROUNDING.value not in parsed.get("task_sequence", []):
            if parsed.get("target_object") is not None:
                parsed["target_object"] = None

        # Always attach correct raw_query
        parsed["raw_query"] = raw_query

        # Step 7: Check for missing fields and backfill from keyword classifier
        backfilled: List[str] = []
        fallback_obj: Optional[InterpretedQuery] = None

        def get_fallback() -> InterpretedQuery:
            nonlocal fallback_obj
            if fallback_obj is None:
                fallback_obj = classify_query_fallback(raw_query)
            return fallback_obj

        # Backfill location if LLM emitted null but keyword classifier found high-confidence location
        if parsed.get("location") is None:
            fb = get_fallback()
            if fb.location is not None and fb.location.lower() in raw_query.lower():
                parsed["location"] = fb.location
                backfilled.append("location")

        # Backfill target_object if grounding task requires it and LLM missed it
        if Task.GROUNDING.value in parsed.get("task_sequence", []):
            if not parsed.get("target_object"):
                fb = get_fallback()
                if fb.target_object:
                    parsed["target_object"] = fb.target_object
                    backfilled.append("target_object")

        # Set default source as LLM (will be updated to HYBRID if backfilled)
        parsed["source"] = Source.LLM.value

        return parsed, backfilled
