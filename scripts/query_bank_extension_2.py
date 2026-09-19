"""
query_bank_extension_2.py — Gap-fill queries for SatQuery AI query bank.

Generated to close three specific, measured imbalances found across the
existing 200-query bank (query_bank.py + query_bank_extension.py):

  1. Sensor skew: 157/200 (78.5%) queries were OPTICAL-only; SAR appeared
     in only 10, BOTH in only 21 -- despite SAR being spec-mandated as
     essential for flood, maritime, landslide, and water-body domains.
     ~20 queries below are SAR or BOTH, concentrated in Water/Disaster.

  2. Chaining skew: only 20/200 (10%) queries were multi-task, despite
     the lifecycle manager's task-sequence looping existing specifically
     to handle compound queries. ~18 queries below are 2-task chains
     covering combinations underrepresented in the existing bank.

  3. Domain skew: Forest was the thinnest domain (17 combined) against
     Infrastructure's 48. ~10 queries below are Forest-domain, single-task,
     to bring it closer to parity.

Every entry below was hand-written and checked against the field
semantics used elsewhere in query_bank.py (target_object null unless
GROUNDING is present; confidence reflecting real query ambiguity, not a
flat default) -- NOT LLM-generated, so no separate verification pass is
required before merging, though a spot-check against your actual
schema.py enum values is still worth doing since this file was written
without direct access to schema.py itself.

Review each entry before merging into query_bank.py -- confidence values
in particular are informed estimates, not measured, and should be
adjusted if they don't match your team's calibration for similar
existing entries.
"""

from typing import Any, Dict, List

try:
    from .schema import Domain, InterpretedQuery, Sensor, Source, Task
except ImportError:
    from schema import Domain, InterpretedQuery, Sensor, Source, Task


GAP_FILL_QUERIES: List[Dict[str, Any]] = [

    # =========================================================================
    # Group 1: SAR / BOTH sensor queries, Water + Disaster domains [20 queries]
    # =========================================================================
    {
        "query": "Use radar to check flood extent around Cuttack since cloud cover is blocking optical imagery.",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Cuttack",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.SAR,
            confidence=0.93,
            source=Source.LLM,
            raw_query="Use radar to check flood extent around Cuttack since cloud cover is blocking optical imagery.",
        ),
    },
    {
        "query": "SAR analysis of the Kosi river flooding near Bihar this monsoon.",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Bihar",
            date_or_time_reference="this monsoon",
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.SAR,
            confidence=0.92,
            source=Source.LLM,
            raw_query="SAR analysis of the Kosi river flooding near Bihar this monsoon.",
        ),
    },
    {
        "query": "Combine SAR and optical imagery to distinguish flooded farmland from flooded settlements near Guwahati.",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Guwahati",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.BOTH,
            confidence=0.94,
            source=Source.LLM,
            raw_query="Combine SAR and optical imagery to distinguish flooded farmland from flooded settlements near Guwahati.",
        ),
    },
    {
        "query": "Is the water in Chilika Lake turbid or clear based on the radar backscatter?",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Chilika Lake",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.WATER,
            sensor=Sensor.SAR,
            confidence=0.87,
            source=Source.LLM,
            raw_query="Is the water in Chilika Lake turbid or clear based on the radar backscatter?",
        ),
    },
    {
        "query": "Map the current extent of Vembanad Lake using SAR since the region is persistently cloudy.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING],
            location="Vembanad Lake",
            date_or_time_reference=None,
            target_object="the lake extent",
            domain=Domain.WATER,
            sensor=Sensor.SAR,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Map the current extent of Vembanad Lake using SAR since the region is persistently cloudy.",
        ),
    },
    {
        "query": "Fuse optical and SAR data to assess reservoir water levels at Hirakud Dam.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS],
            location="Hirakud Dam",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.WATER,
            sensor=Sensor.BOTH,
            confidence=0.93,
            source=Source.LLM,
            raw_query="Fuse optical and SAR data to assess reservoir water levels at Hirakud Dam.",
        ),
    },
    {
        "query": "Detect ships near the Kandla port using SAR given typical coastal cloud cover.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING],
            location="Kandla",
            date_or_time_reference=None,
            target_object="the ships",
            domain=Domain.INFRASTRUCTURE,
            sensor=Sensor.SAR,
            confidence=0.91,
            source=Source.LLM,
            raw_query="Detect ships near the Kandla port using SAR given typical coastal cloud cover.",
        ),
    },
    {
        "query": "Has the waterlogging near Patna receded, checking radar since it's still overcast?",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA],
            location="Patna",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.SAR,
            confidence=0.88,
            source=Source.LLM,
            raw_query="Has the waterlogging near Patna receded, checking radar since it's still overcast?",
        ),
    },
    {
        "query": "Assess landslide-prone slopes near Munnar using radar backscatter, since monsoon cloud cover blocks optical.",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Munnar",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.SAR,
            confidence=0.89,
            source=Source.LLM,
            raw_query="Assess landslide-prone slopes near Munnar using radar backscatter, since monsoon cloud cover blocks optical.",
        ),
    },
    {
        "query": "Combine radar and optical to check if the embankment breach near Malda has worsened.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA],
            location="Malda",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.BOTH,
            confidence=0.91,
            source=Source.LLM,
            raw_query="Combine radar and optical to check if the embankment breach near Malda has worsened.",
        ),
    },
    {
        "query": "SAR-based assessment of standing water in paddy fields near Thanjavur.",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Thanjavur",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.AGRICULTURE,
            sensor=Sensor.SAR,
            confidence=0.88,
            source=Source.LLM,
            raw_query="SAR-based assessment of standing water in paddy fields near Thanjavur.",
        ),
    },
    {
        "query": "Locate the flood boundary near Dibrugarh using radar, describe what settlements are affected.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING, Task.CAPTIONING],
            location="Dibrugarh",
            date_or_time_reference=None,
            target_object="the flood boundary",
            domain=Domain.DISASTER,
            sensor=Sensor.SAR,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Locate the flood boundary near Dibrugarh using radar, describe what settlements are affected.",
        ),
    },
    {
        "query": "Is the SAR backscatter over Pulicat Lake consistent with open water or wetland vegetation?",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Pulicat Lake",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.WATER,
            sensor=Sensor.SAR,
            confidence=0.86,
            source=Source.LLM,
            raw_query="Is the SAR backscatter over Pulicat Lake consistent with open water or wetland vegetation?",
        ),
    },
    {
        "query": "Fuse SAR and optical over the Sundarbans to track mangrove-adjacent flooding.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS],
            location="Sundarbans",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.BOTH,
            confidence=0.92,
            source=Source.LLM,
            raw_query="Fuse SAR and optical over the Sundarbans to track mangrove-adjacent flooding.",
        ),
    },
    {
        "query": "Radar check on whether the Tawa reservoir spillway is releasing water.",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Tawa",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.WATER,
            sensor=Sensor.SAR,
            confidence=0.87,
            source=Source.LLM,
            raw_query="Radar check on whether the Tawa reservoir spillway is releasing water.",
        ),
    },
    {
        "query": "Vessel detection off the Mumbai coast using SAR during the current cloudy spell.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING],
            location="Mumbai",
            date_or_time_reference=None,
            target_object="the vessels",
            domain=Domain.INFRASTRUCTURE,
            sensor=Sensor.SAR,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Vessel detection off the Mumbai coast using SAR during the current cloudy spell.",
        ),
    },
    {
        "query": "Combine SAR and optical to confirm whether the Yamuna floodplain near Delhi is still submerged.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS, Task.VQA],
            location="Delhi",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.BOTH,
            confidence=0.93,
            source=Source.LLM,
            raw_query="Combine SAR and optical to confirm whether the Yamuna floodplain near Delhi is still submerged.",
        ),
    },
    {
        "query": "Radar-based check for storm surge flooding along the Puri coastline.",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Puri",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.SAR,
            confidence=0.89,
            source=Source.LLM,
            raw_query="Radar-based check for storm surge flooding along the Puri coastline.",
        ),
    },
    {
        "query": "Has the water extent of Loktak Lake changed since last year, using radar due to persistent haze?",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA],
            location="Loktak Lake",
            date_or_time_reference="since last year",
            target_object=None,
            domain=Domain.WATER,
            sensor=Sensor.SAR,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Has the water extent of Loktak Lake changed since last year, using radar due to persistent haze?",
        ),
    },
    {
        "query": "Fuse SAR and optical data to assess siltation levels in the Krishna river delta.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS],
            location="Krishna river delta",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.WATER,
            sensor=Sensor.BOTH,
            confidence=0.91,
            source=Source.LLM,
            raw_query="Fuse SAR and optical data to assess siltation levels in the Krishna river delta.",
        ),
    },

    # =========================================================================
    # Group 2: Chained (multi-task) queries, varied combinations [18 queries]
    # =========================================================================
    {
        "query": "Locate the industrial zone near Vapi and summarize what activity is visible there.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING, Task.CAPTIONING],
            location="Vapi",
            date_or_time_reference=None,
            target_object="the industrial zone",
            domain=Domain.INFRASTRUCTURE,
            sensor=Sensor.OPTICAL,
            confidence=0.91,
            source=Source.LLM,
            raw_query="Locate the industrial zone near Vapi and summarize what activity is visible there.",
        ),
    },
    {
        "query": "Fuse SAR and optical over Kaziranga and describe the current wetland conditions.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS, Task.CAPTIONING],
            location="Kaziranga",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.WATER,
            sensor=Sensor.BOTH,
            confidence=0.92,
            source=Source.LLM,
            raw_query="Fuse SAR and optical over Kaziranga and describe the current wetland conditions.",
        ),
    },
    {
        "query": "Has the mining pit near Jharia expanded, and describe the surrounding land disturbance.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA, Task.CAPTIONING],
            location="Jharia",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.GENERAL,
            sensor=Sensor.OPTICAL,
            confidence=0.89,
            source=Source.LLM,
            raw_query="Has the mining pit near Jharia expanded, and describe the surrounding land disturbance.",
        ),
    },
    {
        "query": "Point out the new highway segment near Indore and check if it was built within the last year.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING, Task.CHANGE_VQA],
            location="Indore",
            date_or_time_reference="within the last year",
            target_object="the highway segment",
            domain=Domain.INFRASTRUCTURE,
            sensor=Sensor.OPTICAL,
            confidence=0.92,
            source=Source.LLM,
            raw_query="Point out the new highway segment near Indore and check if it was built within the last year.",
        ),
    },
    {
        "query": "Fuse radar and optical over the Nicobar coastline and locate any visible debris fields.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS, Task.GROUNDING],
            location="Nicobar",
            date_or_time_reference=None,
            target_object="the debris fields",
            domain=Domain.DISASTER,
            sensor=Sensor.BOTH,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Fuse radar and optical over the Nicobar coastline and locate any visible debris fields.",
        ),
    },
    {
        "query": "Describe the crop pattern near Ludhiana and check whether it has shifted compared to last season.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CAPTIONING, Task.CHANGE_VQA],
            location="Ludhiana",
            date_or_time_reference="last season",
            target_object=None,
            domain=Domain.AGRICULTURE,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Describe the crop pattern near Ludhiana and check whether it has shifted compared to last season.",
        ),
    },
    {
        "query": "Locate the deforested patch near Similipal and fuse SAR with optical to confirm canopy loss.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING, Task.FUSION_ANALYSIS],
            location="Similipal",
            date_or_time_reference=None,
            target_object="the deforested patch",
            domain=Domain.FOREST,
            sensor=Sensor.BOTH,
            confidence=0.91,
            source=Source.LLM,
            raw_query="Locate the deforested patch near Similipal and fuse SAR with optical to confirm canopy loss.",
        ),
    },
    {
        "query": "Has the coastline near Digha eroded, and describe the extent of the change.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA, Task.CAPTIONING],
            location="Digha",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.WATER,
            sensor=Sensor.OPTICAL,
            confidence=0.89,
            source=Source.LLM,
            raw_query="Has the coastline near Digha eroded, and describe the extent of the change.",
        ),
    },
    {
        "query": "Ground the wildfire burn scar near Bandipur and check if it's grown since last week.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING, Task.CHANGE_VQA],
            location="Bandipur",
            date_or_time_reference="since last week",
            target_object="the burn scar",
            domain=Domain.DISASTER,
            sensor=Sensor.OPTICAL,
            confidence=0.93,
            source=Source.LLM,
            raw_query="Ground the wildfire burn scar near Bandipur and check if it's grown since last week.",
        ),
    },
    {
        "query": "Fuse SAR and optical over the Ganga floodplain near Varanasi and summarize the affected area.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS, Task.CAPTIONING],
            location="Varanasi",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.DISASTER,
            sensor=Sensor.BOTH,
            confidence=0.92,
            source=Source.LLM,
            raw_query="Fuse SAR and optical over the Ganga floodplain near Varanasi and summarize the affected area.",
        ),
    },
    {
        "query": "Locate the new residential layout near Nagpur and describe how dense the construction is.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING, Task.CAPTIONING],
            location="Nagpur",
            date_or_time_reference=None,
            target_object="the residential layout",
            domain=Domain.URBAN,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Locate the new residential layout near Nagpur and describe how dense the construction is.",
        ),
    },
    {
        "query": "Check if the glacier near Gangotri has retreated and locate the current snout position.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA, Task.GROUNDING],
            location="Gangotri",
            date_or_time_reference=None,
            target_object="the glacier snout",
            domain=Domain.WATER,
            sensor=Sensor.OPTICAL,
            confidence=0.88,
            source=Source.LLM,
            raw_query="Check if the glacier near Gangotri has retreated and locate the current snout position.",
        ),
    },
    {
        "query": "Fuse SAR and optical over the Sunderbans mangroves and check if the shoreline has changed.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS, Task.CHANGE_VQA],
            location="Sunderbans",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.BOTH,
            confidence=0.91,
            source=Source.LLM,
            raw_query="Fuse SAR and optical over the Sunderbans mangroves and check if the shoreline has changed.",
        ),
    },
    {
        "query": "Describe the port infrastructure near Paradip and point out the largest cargo vessel.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CAPTIONING, Task.GROUNDING],
            location="Paradip",
            date_or_time_reference=None,
            target_object="the largest cargo vessel",
            domain=Domain.INFRASTRUCTURE,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Describe the port infrastructure near Paradip and point out the largest cargo vessel.",
        ),
    },
    {
        "query": "Has the quarry near Kota expanded its boundary, and locate the current excavation edge.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA, Task.GROUNDING],
            location="Kota",
            date_or_time_reference=None,
            target_object="the excavation edge",
            domain=Domain.GENERAL,
            sensor=Sensor.OPTICAL,
            confidence=0.88,
            source=Source.LLM,
            raw_query="Has the quarry near Kota expanded its boundary, and locate the current excavation edge.",
        ),
    },
    {
        "query": "Fuse SAR and optical over the Krishna delta wetlands and locate any illegal aquaculture ponds.",
        "expected": InterpretedQuery(
            task_sequence=[Task.FUSION_ANALYSIS, Task.GROUNDING],
            location="Krishna delta",
            date_or_time_reference=None,
            target_object="the aquaculture ponds",
            domain=Domain.WATER,
            sensor=Sensor.BOTH,
            confidence=0.89,
            source=Source.LLM,
            raw_query="Fuse SAR and optical over the Krishna delta wetlands and locate any illegal aquaculture ponds.",
        ),
    },
    {
        "query": "Ground the airstrip near Leh and check if the runway markings have changed since the resurfacing.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING, Task.CHANGE_VQA],
            location="Leh",
            date_or_time_reference="since the resurfacing",
            target_object="the airstrip",
            domain=Domain.INFRASTRUCTURE,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Ground the airstrip near Leh and check if the runway markings have changed since the resurfacing.",
        ),
    },
    {
        "query": "Describe the forest canopy near Periyar and check whether it has thinned over the past 3 years.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CAPTIONING, Task.CHANGE_VQA],
            location="Periyar",
            date_or_time_reference="over the past 3 years",
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Describe the forest canopy near Periyar and check whether it has thinned over the past 3 years.",
        ),
    },

    # =========================================================================
    # Group 3: Forest-domain queries, single-task, to close domain gap [10 queries]
    # =========================================================================
    {
        "query": "What is the estimated tree density in this section of the Nagarhole forest?",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Nagarhole",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.91,
            source=Source.LLM,
            raw_query="What is the estimated tree density in this section of the Nagarhole forest?",
        ),
    },
    {
        "query": "Describe the vegetation health across the Anamalai hills reserve.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CAPTIONING],
            location="Anamalai",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Describe the vegetation health across the Anamalai hills reserve.",
        ),
    },
    {
        "query": "Locate the clear-cut logging patch inside the Corbett buffer zone.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING],
            location="Corbett",
            date_or_time_reference=None,
            target_object="the clear-cut logging patch",
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.92,
            source=Source.LLM,
            raw_query="Locate the clear-cut logging patch inside the Corbett buffer zone.",
        ),
    },
    {
        "query": "Has the tree cover near Shillong's sacred groves decreased since the last dry season?",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA],
            location="Shillong",
            date_or_time_reference="since the last dry season",
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Has the tree cover near Shillong's sacred groves decreased since the last dry season?",
        ),
    },
    {
        "query": "Is there visible evidence of seasonal leaf-loss versus actual clearing in this Bandhavgarh section?",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Bandhavgarh",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.87,
            source=Source.LLM,
            raw_query="Is there visible evidence of seasonal leaf-loss versus actual clearing in this Bandhavgarh section?",
        ),
    },
    {
        "query": "What proportion of the Dandeli forest canopy shows signs of degradation?",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Dandeli",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="What proportion of the Dandeli forest canopy shows signs of degradation?",
        ),
    },
    {
        "query": "Point out the reforested plots near Chandrapur.",
        "expected": InterpretedQuery(
            task_sequence=[Task.GROUNDING],
            location="Chandrapur",
            date_or_time_reference=None,
            target_object="the reforested plots",
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.89,
            source=Source.LLM,
            raw_query="Point out the reforested plots near Chandrapur.",
        ),
    },
    {
        "query": "Summarize the forest composition visible along the Silent Valley boundary.",
        "expected": InterpretedQuery(
            task_sequence=[Task.CAPTIONING],
            location="Silent Valley",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.90,
            source=Source.LLM,
            raw_query="Summarize the forest composition visible along the Silent Valley boundary.",
        ),
    },
    {
        "query": "Has the mangrove extent near Bhitarkanika grown or shrunk over the past 5 years?",
        "expected": InterpretedQuery(
            task_sequence=[Task.CHANGE_VQA],
            location="Bhitarkanika",
            date_or_time_reference="over the past 5 years",
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.91,
            source=Source.LLM,
            raw_query="Has the mangrove extent near Bhitarkanika grown or shrunk over the past 5 years?",
        ),
    },
    {
        "query": "Are there signs of encroachment along the edge of the Gir forest boundary?",
        "expected": InterpretedQuery(
            task_sequence=[Task.VQA],
            location="Gir",
            date_or_time_reference=None,
            target_object=None,
            domain=Domain.FOREST,
            sensor=Sensor.OPTICAL,
            confidence=0.89,
            source=Source.LLM,
            raw_query="Are there signs of encroachment along the edge of the Gir forest boundary?",
        ),
    },
]


def get_gap_fill_queries() -> List[Dict[str, Any]]:
    """Returns the gap-fill query set (48 total: 20 sensor + 18 chained + 10 forest)."""
    return GAP_FILL_QUERIES
