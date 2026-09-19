"""
schema.py — the JSON contract for the SatQuery AI query interpreter.

This is the single source of truth for what an "interpreted query" looks like.
Every other piece of the interpreter (fallback classifier, LLM prompt output,
validation, the test harness) must produce or consume objects matching this
shape. If you change a field here, update the prompt template, the fallback
classifier, and the query bank labels in the same pass — they will silently
drift out of sync otherwise.
"""

from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Controlled vocabularies — these are the only valid values for their fields.
# Keep these in sync with the task list in the problem statement's mandatory
# functional scope section.
# ---------------------------------------------------------------------------

class Task(str, Enum):
    VQA = "vqa"
    CAPTIONING = "captioning"
    GROUNDING = "grounding"
    CHANGE_VQA = "change_vqa"
    FUSION_ANALYSIS = "fusion_analysis"
    GEOCHAT = "geochat"


class Sensor(str, Enum):
    OPTICAL = "optical"
    SAR = "sar"
    BOTH = "both"
    UNSPECIFIED = "unspecified"


class Domain(str, Enum):
    URBAN = "urban"
    WATER = "water"
    AGRICULTURE = "agriculture"
    FOREST = "forest"
    DISASTER = "disaster"
    INFRASTRUCTURE = "infrastructure"
    GENERAL = "general"  # fallback when the query doesn't fit a specific domain


class Source(str, Enum):
    LLM = "llm"                    # Qwen produced valid JSON directly
    KEYWORD_FALLBACK = "keyword_fallback"  # Qwen failed/unavailable, regex classifier ran instead
    HYBRID = "hybrid"               # Qwen partially succeeded, some fields backfilled by fallback
    USER_CLARIFIED = "user_clarified"  # A person answered a clarifying question; result is remembered


# ---------------------------------------------------------------------------
# The interpreted query itself
# ---------------------------------------------------------------------------

class InterpretedQuery(BaseModel):
    task_sequence: List[Task] = Field(
        ...,
        min_length=1,
        description=(
            "Ordered list of tasks this query requires. Most queries have exactly "
            "one task. A query like 'highlight the water body and tell me if it "
            "changed' requires two, in order: [grounding, change_vqa]."
        ),
    )
    location: Optional[str] = Field(
        default=None,
        description="Raw place name extracted from the query text, e.g. 'Chennai'. "
                     "Null if no location is mentioned (e.g. user is uploading images directly).",
    )
    date_or_time_reference: Optional[str] = Field(
        default=None,
        description="Raw date/time text as the user phrased it, e.g. 'before and after "
                     "monsoon' or '2023 vs 2024'. Not parsed into actual dates here — "
                     "that's a downstream concern, this field just flags that a temporal "
                     "signal exists and preserves the user's original phrasing.",
    )
    target_object: Optional[str] = Field(
        default=None,
        description="For grounding tasks: the object/region to highlight, e.g. "
                     "'the water body', 'the built-up area'. Null for non-grounding tasks.",
    )
    domain: Domain = Field(
        default=Domain.GENERAL,
        description="Coarse subject-matter category, used for query bank labeling and "
                     "for routing/logging, not for model dispatch directly.",
    )
    sensor: Sensor = Field(
        default=Sensor.UNSPECIFIED,
        description="Which sensor modality the query implies. 'both' should correlate "
                     "with fusion_analysis being in task_sequence, but the two are validated "
                     "independently — see cross-field check below.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How confident the interpreter is in this result. LLM-sourced results "
                     "should self-report this; keyword_fallback results should be capped "
                     "low (e.g. <= 0.5) since regex matching is a weaker signal than the LLM.",
    )
    source: Source = Field(
        ...,
        description="Which path produced this result — see Source enum above.",
    )
    raw_query: str = Field(
        ...,
        description="The original, unmodified user query text. Always keep this attached "
                     "to the parsed result for debugging and for the test harness to log "
                     "against failure cases.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="True when confidence is too low to trust this result automatically. "
                     "The interpreter still returns its best guess (never blocks the "
                     "pipeline), but the caller (UI/orchestrator) should consider asking "
                     "the user a clarifying question and calling "
                     "QueryInterpreter.apply_user_clarification() with the answer, which "
                     "permanently remembers the correction for this query.",
    )

    @field_validator("task_sequence")
    @classmethod
    def task_sequence_no_duplicates(cls, v: List[Task]) -> List[Task]:
        if len(v) != len(set(v)):
            raise ValueError("task_sequence contains duplicate tasks")
        return v

    @field_validator("confidence")
    @classmethod
    def fallback_confidence_capped(cls, v: float, info) -> float:
        # Note: cross-field validation (checking `source` alongside `confidence`)
        # requires model_validator instead of field_validator in pydantic v2 if you
        # want this enforced automatically. Left as a manual convention for now —
        # enforce it in the fallback classifier itself (cap at 0.5) rather than here,
        # to keep this validator simple. Revisit if source/confidence drift happens
        # in practice once real data comes in.
        return v


# ---------------------------------------------------------------------------
# Example valid objects — useful for the test harness and for sanity-checking
# the prompt template's few-shot examples against the real schema.
# ---------------------------------------------------------------------------

EXAMPLE_SINGLE_TASK = InterpretedQuery(
    task_sequence=[Task.VQA],
    location="Chennai",
    date_or_time_reference=None,
    target_object=None,
    domain=Domain.URBAN,
    sensor=Sensor.OPTICAL,
    confidence=0.92,
    source=Source.LLM,
    raw_query="What does the land cover look like around Chennai?",
)

EXAMPLE_CHAINED_TASK = InterpretedQuery(
    task_sequence=[Task.GROUNDING, Task.CHANGE_VQA],
    location=None,
    date_or_time_reference="before and after monsoon",
    target_object="the water body",
    domain=Domain.WATER,
    sensor=Sensor.OPTICAL,
    confidence=0.81,
    source=Source.LLM,
    raw_query="Highlight the water body and tell me if it changed before and after monsoon.",
)

EXAMPLE_FALLBACK = InterpretedQuery(
    task_sequence=[Task.VQA],
    location=None,
    date_or_time_reference=None,
    target_object=None,
    domain=Domain.GENERAL,
    sensor=Sensor.UNSPECIFIED,
    confidence=0.3,
    source=Source.KEYWORD_FALLBACK,
    raw_query="tell me about this",
)


if __name__ == "__main__":
    # Quick sanity check when running this file directly.
    for example in (EXAMPLE_SINGLE_TASK, EXAMPLE_CHAINED_TASK, EXAMPLE_FALLBACK):
        print(example.model_dump_json(indent=2))
        print("---")
