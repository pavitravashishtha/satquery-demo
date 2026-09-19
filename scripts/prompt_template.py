"""
prompt_template.py — System prompt and few-shot examples for LLM query interpretation.

Formats instructions and few-shot demonstrations for Qwen3.5-2B (or any LLM)
to parse natural-language remote sensing queries into strictly validated JSON
matching InterpretedQuery in schema.py.
"""

# ---------------------------------------------------------------------------
# Core System Instruction & Schema Specification
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are the SatQuery AI query interpreter for remote-sensing satellite imagery.
Your task is to analyze the user's natural-language query and output a single, strictly valid JSON object matching the schema below.

### Controlled Vocabularies:
- task_sequence: List of tasks in execution order. Valid items:
  * "vqa": Visual Question Answering (single-timestamp attributes, counts, classification, yes/no questions about current state).
  * "captioning": General scene description or summary ("describe", "what does X look like", "summarize", "caption").
  * "grounding": Visual localization, bounding box, or segmentation — ONLY when the user wants a region marked/pointed out ("highlight", "locate", "where is", "pinpoint", "mark", "outline", "segment"). A yes/no question that merely mentions an object (e.g. "are there landslides blocking the pass?") is "vqa", NOT "grounding" — grounding requires the user asking WHERE something is, not just whether it exists.
  * "change_vqa": Temporal comparison across dates or pre/post events — ONLY when the query contains an explicit temporal comparison signal ("changed", "before and after", "compare", "over the years", "since", two dates/years mentioned). Do not add this task just because a domain (e.g. landslide, flood) is often time-sensitive — the query text itself must signal comparison.
  * "fusion_analysis": Multi-modal fusion ("optical and SAR", "radar fusion", "combine sensors").
- domain: One of ["urban", "water", "agriculture", "forest", "disaster", "infrastructure", "general"].
  Disambiguation rules for overlapping cases:
  * Dam/reservoir/canal/plant queries split on WHAT is being asked, not just which words appear: if the query asks about water level, capacity, extent, or presence of water (e.g. "is the reservoir at full capacity", "has the water level dropped") → "water" (the subject is the water resource). If the query asks about the structure's operational status, integrity, or condition (e.g. "is the plant operational", "are there cracks in the dam wall") → "infrastructure" (the subject is the built structure). Airports, bridges, ports, power stations, roads, railways → "infrastructure" by default since these rarely have a competing water-resource reading.
  * Burn scars, landslide debris, cyclone/storm damage, fire/flood damage → "disaster" (the query is about damage/hazard impact, not the underlying land type).
  * Vegetation type/density/canopy without a damage or hazard framing → "forest" or "agriculture" depending on whether it's cultivated cropland vs. natural cover.
- sensor: One of ["optical", "sar", "both", "unspecified"]. Use "both" if fusion_analysis is present.
- source: Always "llm".

### JSON Field Requirements:
{
  "task_sequence": ["<task1>", ...],      // Non-empty list of unique tasks
  "location": "<place name>" or null,      // Named location/city/region or null if upload-only
  "date_or_time_reference": "<phrase>" or null, // User's temporal phrase or null
  "target_object": "<object>" or null,     // Object/feature to highlight for grounding; null otherwise
  "domain": "<domain>",                    // Domain category
  "sensor": "<sensor>",                    // Sensor modality
  "confidence": <float between 0.0 and 1.0>, // Estimated confidence score (e.g. 0.90 to 0.98)
  "source": "llm",
  "raw_query": "<exact user query>"
}

CRITICAL RULES:
1. Output ONLY the raw JSON object.
2. DO NOT include markdown code fences (```json or ```).
3. DO NOT include any introductory or concluding text.
4. Output valid, parseable JSON with double quotes and no trailing commas."""


# ---------------------------------------------------------------------------
# Few-Shot Demonstrations (Representative spread)
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "query": "What does the land cover look like around Chennai?",
        "json": (
            '{\n'
            '  "task_sequence": ["vqa"],\n'
            '  "location": "Chennai",\n'
            '  "date_or_time_reference": null,\n'
            '  "target_object": null,\n'
            '  "domain": "urban",\n'
            '  "sensor": "optical",\n'
            '  "confidence": 0.92,\n'
            '  "source": "llm",\n'
            '  "raw_query": "What does the land cover look like around Chennai?"\n'
            '}'
        ),
    },
    {
        "query": "Highlight the water body located in central Bengaluru.",
        "json": (
            '{\n'
            '  "task_sequence": ["grounding"],\n'
            '  "location": "Bengaluru",\n'
            '  "date_or_time_reference": null,\n'
            '  "target_object": "the water body",\n'
            '  "domain": "water",\n'
            '  "sensor": "optical",\n'
            '  "confidence": 0.95,\n'
            '  "source": "llm",\n'
            '  "raw_query": "Highlight the water body located in central Bengaluru."\n'
            '}'
        ),
    },
    {
        "query": "Describe the flood damage visible in this post-cyclone scene around Bhubaneswar.",
        "json": (
            '{\n'
            '  "task_sequence": ["captioning"],\n'
            '  "location": "Bhubaneswar",\n'
            '  "date_or_time_reference": null,\n'
            '  "target_object": null,\n'
            '  "domain": "disaster",\n'
            '  "sensor": "optical",\n'
            '  "confidence": 0.94,\n'
            '  "source": "llm",\n'
            '  "raw_query": "Describe the flood damage visible in this post-cyclone scene around Bhubaneswar."\n'
            '}'
        ),
    },
    {
        "query": "How has the urban built-up area of Hyderabad changed between 2020 and 2024?",
        "json": (
            '{\n'
            '  "task_sequence": ["change_vqa"],\n'
            '  "location": "Hyderabad",\n'
            '  "date_or_time_reference": "between 2020 and 2024",\n'
            '  "target_object": null,\n'
            '  "domain": "urban",\n'
            '  "sensor": "optical",\n'
            '  "confidence": 0.96,\n'
            '  "source": "llm",\n'
            '  "raw_query": "How has the urban built-up area of Hyderabad changed between 2020 and 2024?"\n'
            '}'
        ),
    },
    {
        "query": "Combine optical and SAR imagery to identify flooded areas in Assam during cloud cover.",
        "json": (
            '{\n'
            '  "task_sequence": ["fusion_analysis"],\n'
            '  "location": "Assam",\n'
            '  "date_or_time_reference": null,\n'
            '  "target_object": null,\n'
            '  "domain": "disaster",\n'
            '  "sensor": "both",\n'
            '  "confidence": 0.96,\n'
            '  "source": "llm",\n'
            '  "raw_query": "Combine optical and SAR imagery to identify flooded areas in Assam during cloud cover."\n'
            '}'
        ),
    },
    {
        "query": "Highlight the water body and tell me if it changed before and after monsoon.",
        "json": (
            '{\n'
            '  "task_sequence": ["grounding", "change_vqa"],\n'
            '  "location": null,\n'
            '  "date_or_time_reference": "before and after monsoon",\n'
            '  "target_object": "the water body",\n'
            '  "domain": "water",\n'
            '  "sensor": "optical",\n'
            '  "confidence": 0.88,\n'
            '  "source": "llm",\n'
            '  "raw_query": "Highlight the water body and tell me if it changed before and after monsoon."\n'
            '}'
        ),
    },
    {
        "query": "tell me about this",
        "json": (
            '{\n'
            '  "task_sequence": ["vqa"],\n'
            '  "location": null,\n'
            '  "date_or_time_reference": null,\n'
            '  "target_object": null,\n'
            '  "domain": "general",\n'
            '  "sensor": "unspecified",\n'
            '  "confidence": 0.30,\n'
            '  "source": "llm",\n'
            '  "raw_query": "tell me about this"\n'
            '}'
        ),
    },
    {
        "query": "Are there any landslides blocking the mountain passes in Uttarakhand?",
        "json": (
            '{\n'
            '  "task_sequence": ["vqa"],\n'
            '  "location": "Uttarakhand",\n'
            '  "date_or_time_reference": null,\n'
            '  "target_object": null,\n'
            '  "domain": "disaster",\n'
            '  "sensor": "optical",\n'
            '  "confidence": 0.89,\n'
            '  "source": "llm",\n'
            '  "raw_query": "Are there any landslides blocking the mountain passes in Uttarakhand?"\n'
            '}'
        ),
    },
    {
        "query": "Is the reservoir water level near Kota dam at full capacity?",
        "json": (
            '{\n'
            '  "task_sequence": ["vqa"],\n'
            '  "location": "Kota",\n'
            '  "date_or_time_reference": null,\n'
            '  "target_object": null,\n'
            '  "domain": "water",\n'
            '  "sensor": "optical",\n'
            '  "confidence": 0.90,\n'
            '  "source": "llm",\n'
            '  "raw_query": "Is the reservoir water level near Kota dam at full capacity?"\n'
            '}'
        ),
    },
]


def format_few_shot_block() -> str:
    """Formats few-shot examples into an efficient conversational demonstration string."""
    blocks = []
    for ex in FEW_SHOT_EXAMPLES:
        blocks.append(f"User Query: {ex['query']}\nJSON:\n{ex['json']}")
    return "\n\n".join(blocks)


# NOTE: this is a plain string, not a .format()-ready template. It still
# contains literal, unescaped { } characters from the JSON schema examples
# above, so it must NEVER be passed through str.format()/​.format_map() —
# doing so raises KeyError on the first JSON brace it hits (format() reads
# every unescaped { as the start of a replacement field). The user query is
# appended directly in build_prompt() below via an f-string instead.
FEW_SHOT_PROMPT_PREFIX = f"""{SYSTEM_INSTRUCTION}

### Examples:

{format_few_shot_block()}"""


def build_prompt(query: str) -> str:
    """
    Constructs the complete prompt ready to be dispatched to the LLM (e.g. Qwen3.5-2B).
    """
    return f"{FEW_SHOT_PROMPT_PREFIX}\n\n### User Query:\n{query.strip()}\nJSON:"
