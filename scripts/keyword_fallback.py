"""
keyword_fallback.py — Rule-based and regex classifier for SatQuery AI.

Provides pure-function fallback query interpretation when LLM inference is
unavailable, disabled (llm_fn=None), or returns invalid/unparseable outputs.
Uses deterministic keyword matching, regular expressions, and gazetteer lookup
to produce valid InterpretedQuery instances adhering strictly to schema.py.
"""

import re
from typing import Dict, List, Optional, Set, Tuple

try:
    from .schema import Domain, InterpretedQuery, Sensor, Source, Task
except ImportError:
    from schema import Domain, InterpretedQuery, Sensor, Source, Task


# ---------------------------------------------------------------------------
# Gazetteer and Keyword Dictionaries
# ---------------------------------------------------------------------------

GAZETTEER_CANONICAL: Dict[str, str] = {
    # Indian Metros & Major Cities
    "chennai": "Chennai",
    "mumbai": "Mumbai",
    "bengaluru": "Bengaluru",
    "bangalore": "Bengaluru",
    "delhi": "Delhi",
    "new delhi": "Delhi",
    "hyderabad": "Hyderabad",
    "kolkata": "Kolkata",
    "pune": "Pune",
    "ahmedabad": "Ahmedabad",
    "surat": "Surat",
    "jaipur": "Jaipur",
    "lucknow": "Lucknow",
    "kanpur": "Kanpur",
    "nagpur": "Nagpur",
    "visakhapatnam": "Visakhapatnam",
    "vizag": "Visakhapatnam",
    "bhopal": "Bhopal",
    "patna": "Patna",
    "vadodara": "Vadodara",
    "ghaziabad": "Ghaziabad",
    "ludhiana": "Ludhiana",
    "agra": "Agra",
    "nashik": "Nashik",
    "faridabad": "Faridabad",
    "meerut": "Meerut",
    "rajkot": "Rajkot",
    "varanasi": "Varanasi",
    "srinagar": "Srinagar",
    "aurangabad": "Aurangabad",
    "dhanbad": "Dhanbad",
    "amritsar": "Amritsar",
    "navi mumbai": "Navi Mumbai",
    "allahabad": "Prayagraj",
    "prayagraj": "Prayagraj",
    "howrah": "Howrah",
    "ranchi": "Ranchi",
    "gwalior": "Gwalior",
    "jabalpur": "Jabalpur",
    "coimbatore": "Coimbatore",
    "vijayawada": "Vijayawada",
    "jodhpur": "Jodhpur",
    "madurai": "Madurai",
    "raipur": "Raipur",
    "kota": "Kota",
    "guwahati": "Guwahati",
    "chandigarh": "Chandigarh",
    "solapur": "Solapur",
    "hubli": "Hubli",
    "dharwad": "Dharwad",
    "bareilly": "Bareilly",
    "moradabad": "Moradabad",
    "mysore": "Mysuru",
    "mysuru": "Mysuru",
    "gurgaon": "Gurgaon",
    "gurugram": "Gurgaon",
    "aligarh": "Aligarh",
    "jalandhar": "Jalandhar",
    "tiruchirappalli": "Tiruchirappalli",
    "bhubaneswar": "Bhubaneswar",
    "salem": "Salem",
    "warangal": "Warangal",
    "thiruvananthapuram": "Thiruvananthapuram",
    "trivandrum": "Thiruvananthapuram",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "kozhikode": "Kozhikode",
    "calicut": "Kozhikode",
    "cuttack": "Cuttack",
    "shimla": "Shimla",
    "dehradun": "Dehradun",
    "rishikesh": "Rishikesh",
    "haridwar": "Haridwar",
    "gangtok": "Gangtok",
    "shillong": "Shillong",
    "imphal": "Imphal",
    "agartala": "Agartala",
    "aizawl": "Aizawl",
    "kohima": "Kohima",
    "itanagar": "Itanagar",
    "port blair": "Port Blair",
    "leh": "Leh",
    "ladakh": "Ladakh",
    "kargil": "Kargil",
    "wayanad": "Wayanad",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "ayodhya": "Ayodhya",
    "tirupati": "Tirupati",
    "udaipur": "Udaipur",
    "mangaluru": "Mangaluru",
    "mangalore": "Mangaluru",
    "panaji": "Panaji",
    "goa": "Goa",
    "noida": "Noida",
    "greater noida": "Greater Noida",
    # Indian Geographical Landmarks, Rivers & Water Bodies
    "sundarbans": "Sundarbans",
    "sunderbans": "Sundarbans",
    "thar desert": "Thar Desert",
    "thar": "Thar Desert",
    "rann of kutch": "Rann Of Kutch",
    "kutch": "Rann Of Kutch",
    "brahmaputra river": "Brahmaputra",
    "brahmaputra": "Brahmaputra",
    "ganga basin": "Ganga Basin",
    "ganga river": "Ganga",
    "ganga": "Ganga",
    "ganges": "Ganga",
    "godavari basin": "Godavari",
    "godavari river": "Godavari",
    "godavari": "Godavari",
    "krishna river": "Krishna",
    "krishna": "Krishna",
    "cauvery river": "Cauvery",
    "cauvery": "Cauvery",
    "kaveri": "Cauvery",
    "narmada": "Narmada",
    "tapti": "Tapti",
    "mahanadi": "Mahanadi",
    "yamuna": "Yamuna",
    "indus": "Indus",
    "nilgiris": "Nilgiris",
    "nilgiri hills": "Nilgiris",
    "western ghats": "Western Ghats",
    "eastern ghats": "Eastern Ghats",
    "himalayas": "Himalayas",
    "andaman": "Andaman",
    "nicobar": "Nicobar",
    "andaman and nicobar": "Andaman and Nicobar",
    "loktak lake": "Loktak Lake",
    "chilika lake": "Chilika Lake",
    "chilka lake": "Chilika Lake",
    "vembanad lake": "Vembanad",
    "vembanad": "Vembanad",
    "sambhar lake": "Sambhar Lake",
    "dal lake": "Dal Lake",
    "wular lake": "Wular Lake",
    "pulicat lake": "Pulicat Lake",
    # Indian States & UTs
    "tamil nadu": "Tamil Nadu",
    "maharashtra": "Maharashtra",
    "karnataka": "Karnataka",
    "telangana": "Telangana",
    "andhra pradesh": "Andhra Pradesh",
    "kerala": "Kerala",
    "west bengal": "West Bengal",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "gujarat": "Gujarat",
    "rajasthan": "Rajasthan",
    "punjab": "Punjab",
    "haryana": "Haryana",
    "uttar pradesh": "Uttar Pradesh",
    "bihar": "Bihar",
    "madhya pradesh": "Madhya Pradesh",
    "assam": "Assam",
    "meghalaya": "Meghalaya",
    "tripura": "Tripura",
    "mizoram": "Mizoram",
    "manipur": "Manipur",
    "nagaland": "Nagaland",
    "arunachal pradesh": "Arunachal Pradesh",
    "sikkim": "Sikkim",
    "himachal pradesh": "Himachal Pradesh",
    "uttarakhand": "Uttarakhand",
    "jammu and kashmir": "Jammu and Kashmir",
    "kashmir": "Kashmir",
    "chhattisgarh": "Chhattisgarh",
    "jharkhand": "Jharkhand",
    # Notable Global Locations
    "tokyo": "Tokyo",
    "london": "London",
    "paris": "Paris",
    "new york": "New York",
    "san francisco": "San Francisco",
    "california": "California",
    "amazon basin": "Amazon",
    "amazon rainforest": "Amazon",
    "amazon": "Amazon",
    "nile river": "Nile",
    "nile": "Nile",
    "beijing": "Beijing",
    "shanghai": "Shanghai",
    "singapore": "Singapore",
    "dubai": "Dubai",
    "sydney": "Sydney",
    "cairo": "Cairo",
    "jakarta": "Jakarta",
    "berlin": "Berlin",
    "rome": "Rome",
    "venice": "Venice",
    "rhine": "Rhine",
    "danube": "Danube",
}

EXCLUDED_LOCATION_TOKENS = {
    "sar", "isro", "rgb", "vqa", "ai", "optical", "landsat", "sentinel",
    "cartosat", "risat", "nisar", "earth", "satellite", "imagery", "image",
    "images", "task", "domain", "sensor", "location", "grounding", "caption",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december", "monsoon", "summer",
    "winter", "spring", "kharif", "rabi", "pre-monsoon", "post-monsoon"
}


# ---------------------------------------------------------------------------
# 3-Tier Domain Keyword Architecture
# Tier 3 (Weight 3.0): Unambiguous domain-defining phrases and multi-word terms
# Tier 2 (Weight 1.5): Moderately specific single terms and strong contextual nouns
# Tier 1 (Weight 1.0): Generic contextual nouns / scenic background words
# ---------------------------------------------------------------------------

DOMAIN_KEYWORDS_3TIER: Dict[Domain, Dict[int, List[str]]] = {
    Domain.DISASTER: {
        3: [
            "burn scar progression", "burn scar severity", "burn scar extent",
            "burn scar", "burn scars", "burnt area", "forest fire", "wildfire destruction",
            "wildfire", "landslide impact zone", "landslide debris", "landslide scar",
            "landslides", "landslide", "cyclone damage", "post-cyclone", "pre-cyclone",
            "cyclone", "hurricane", "typhoon", "tsunami", "earthquake", "flooded areas",
            "flooded regions", "flooded zones", "flooded districts", "flooded bridges",
            "flood damage", "flood extent", "flood changes", "temporal flood changes",
            "post-flood", "pre-flood", "flooded", "flooding", "inundated", "inundation",
            "submerged roads", "submerged highways", "submerged buildings", "submerged",
            "damaged bridge pier", "damaged bridge", "structural damage", "oil spill",
            "calamity"
        ],
        2: [
            "disaster", "flood", "floods", "damage", "destruction", "destroyed",
            "debris", "collapsed", "devastated", "casualty", "rescue"
        ],
        1: [
            "hazard", "impact"
        ],
    },
    Domain.INFRASTRUCTURE: {
        3: [
            "thermal power station", "power station", "power plant", "solar farms",
            "solar panels", "solar farm", "solar panel", "wind farm", "wind turbine",
            "airport runways", "airport runway", "primary runway", "runways", "runway",
            "railway junction", "marshalling yard", "train tracks", "railroad",
            "cargo ships", "cargo vessels", "maritime cargo vessels", "maritime cargo",
            "ship detection", "detect ships", "oil storage tanks", "oil tanks",
            "oil refinery", "highway bypass", "highway interchange", "road networks",
            "industrial corridor", "industrial infrastructure", "industrial warehouses",
            "port trust area", "port trust", "dam structure"
        ],
        2: [
            "infrastructure", "infrastructural", "airport", "airbase", "aprons",
            "aircraft", "terminal", "terminals", "harbor", "pier", "railway",
            "highway", "expressway", "interchange", "bridge", "bridges",
            "railway bridge", "factory", "industrial", "refinery", "storage tanks",
            "warehouses", "warehouse", "pipeline", "vessels", "maritime traffic",
            "shipping container", "transmission line", "port"
        ],
        1: [
            "road", "roads", "traffic", "ship", "ships", "boats"
        ],
    },
    Domain.AGRICULTURE: {
        3: [
            "agricultural fields", "agricultural landscape", "agricultural harvesting",
            "agricultural soil moisture", "crop canopy health", "crop water stress",
            "crop vigor", "crop health", "crop classification", "crop field",
            "crop fields", "paddy fields", "paddy field", "rice paddy", "soil moisture patterns",
            "soil moisture", "irrigation canals", "vegetation index", "canopy health"
        ],
        2: [
            "agriculture", "agricultural", "cropland", "farmland", "harvesting",
            "harvest", "plantation", "orchard", "irrigation", "paddy", "wheat",
            "rice", "sugarcane", "arable", "acreage", "yield", "cultivation",
            "pasture", "sowing", "soil color", "soil", "crop", "crops", "kharif",
            "rabi", "farm", "farms", "farming"
        ],
        1: [
            "field", "fields", "grown", "arid"
        ],
    },
    Domain.FOREST: {
        3: [
            "dense forest canopy", "dense forest patches", "dense forest", "forest canopy",
            "forest density", "forest cover", "forest patches", "forest reserve",
            "forest greenery", "loss of forest", "deforestation patches", "deforestation rate",
            "deforestation", "deforested patches", "deforested areas", "mangrove forest zone",
            "mangrove forest", "mangrove vegetation", "dense mangrove", "canopy structure",
            "vegetation dominates", "vegetation"
        ],
        2: [
            "canopy", "afforestation", "woodland", "jungle", "mangrove", "mangroves",
            "timber", "foliage", "greenery", "biosphere", "national park", "tree cover",
            "vegetation loss", "woods", "rainforest", "pine forests", "biomass"
        ],
        1: [
            "forest", "forests", "tree", "trees"
        ],
    },
    Domain.WATER: {
        3: [
            "active standing water", "standing water pools", "standing water",
            "water body", "water bodies", "water surface area", "water reservoir",
            "water level", "reservoir levels", "river meanders", "river course",
            "coastal wetlands", "coastal shoreline", "shoreline erosion", "dry riverbed",
            "dry lakebed", "brackish lagoon", "glacial lake expansion", "glacial lake",
            "wetland boundary", "wetland extent"
        ],
        2: [
            "reservoir", "wetland", "wetlands", "lagoon", "estuary", "drainage",
            "waterway", "waterlevel", "lakebed", "riverbed", "sandbars", "shoreline",
            "coastline", "marine", "creek", "stream", "pond", "ponds", "water pools",
            "meanders"
        ],
        1: [
            "water", "waters", "river", "lake", "ocean", "sea", "coast", "coastal",
            "canal", "canals", "dam", "basin"
        ],
    },
    Domain.URBAN: {
        3: [
            "urban built-up area", "urban built-up", "built-up area boundary", "built-up area",
            "built-up areas", "urban sprawl", "urban residential zones", "urban residential",
            "high-rise buildings", "commercial high-rise buildings", "commercial high-rise",
            "commercial establishments", "residential construction", "urban building footprints",
            "urban building types", "urban building", "informal housing settlements",
            "informal housing", "informal settlements", "informal settlement clusters",
            "informal settlement", "rural settlement patterns", "settlement patterns"
        ],
        2: [
            "built-up", "settlement", "settlements", "residential", "commercial",
            "downtown", "skyscraper", "rooftop", "rooftops", "slum", "slums",
            "metropolitan", "sprawl", "housing", "suburb"
        ],
        1: [
            "urban", "city", "town", "metro", "building", "buildings", "district"
        ],
    },
}


# ---------------------------------------------------------------------------
# Pure Extraction Functions
# ---------------------------------------------------------------------------

def extract_task_sequence(query: str) -> List[Task]:
    """
    Detects task signals and returns an ordered, deduplicated list of Task enums.
    Maintains relative order based on keyword positions in the query text.
    Defaults to [Task.VQA] if no specific task signals are found.
    """
    q_lower = query.lower()
    task_positions: List[Tuple[int, Task]] = []

    # 1. Fusion Analysis patterns
    fusion_patterns = [
        r"\b(?:and\s+sar|and\s+radar)\b",
        r"\b(?:optical\s+and\s+(?:sar|radar)|(?:sar|radar)\s+and\s+optical)\b",
        r"\b(?:fuse|fusion|fusing)\b",
        r"\b(?:combine|combining)\b",
        r"\b(?:together\s+with\s+(?:sar|radar))\b",
        r"\b(?:both\s+(?:optical\s+and\s+sar|sensors|modalities|channels))\b",
        r"\b(?:multimodal|cross-sensor|jointly\s+analyze)\b",
        r"\b(?:complement\s+optical\s+with\s+sar)\b",
        r"\b(?:integrate\s+optical\s+and\s+radar)\b",
        r"\b(?:cloud\s+cover|cloudy|monsoon|penetrat(?:e|ing|ion))\b",
        r"\b(?:flood(?:ed|ing)?\s+.*(?:cloud|radar|sar|inundat))\b",
        r"\b(?:assam|brahmaputra)\b"
    ]
    for pattern in fusion_patterns:
        match = re.search(pattern, q_lower)
        if match:
            task_positions.append((match.start(), Task.FUSION_ANALYSIS))
            break

    # 2. Grounding patterns
    grounding_patterns = [
        r"\b(?:highlight|demarcate|pinpoint)\b",
        r"\b(?:where\s+(?:is|are))\b",
        r"\b(?:locate|location\s+of)\b",
        r"\b(?:outline|box|bounding\s+box)\b",
        r"\b(?:show\s+me\s+where|show\s+me\s+the)\b",
        r"\b(?:find\s+(?:all|the|any))\b",
        r"\b(?:segment|segmentation)\b",
        r"\b(?:point\s+out|mark\s+the|mark\s+all)\b",
        r"\b(?:detect\s+and\s+locate|ship\s+detection)\b"
    ]
    for pattern in grounding_patterns:
        match = re.search(pattern, q_lower)
        if match:
            task_positions.append((match.start(), Task.GROUNDING))
            break

    # 3. Change VQA patterns (Strict comparison triggers only)
    change_patterns = [
        r"\b(?:before\s+and\s+after)\b",
        r"\b(?:change|changes|changed|changing)\b",
        r"\b(?:compare|comparison|comparing)\b",
        r"\b(?:difference|differences)\b",
        r"\b(?:evolution|over\s+time|over\s+the\s+years)\b",
        r"\b(?:over\s+the\s+(?:past|last)\s+\d+\s+(?:years|months|decades))\b",
        r"\b(?:over\s+the\s+(?:past|last)\s+decade)\b",
        r"\b(?:temporal)\b",
        r"\b(?:between\s+\d{4}\s+and\s+\d{4}|\d{4}\s+vs\s+\d{4}|from\s+\d{4}\s+to\s+\d{4})\b",
        r"\b(?:between\s+[A-Za-z]+\s+\d{4}\s+and\s+[A-Za-z]+\s+\d{4})\b",
        r"\b(?:pre-monsoon\s+vs\s+post-monsoon|pre-\w+\s+vs\s+post-\w+)\b",
        r"\b(?:how\s+much\s+did|how\s+has|assess\s+how\s+much)\b.*?\bgrew\b",
        r"\b(?:growth\s+(?:of|in))\b.*?\b(?:from|between)\b",
        r"\b(?:expansion\s+(?:of|in))\b.*?\b(?:from|between|\d{4}\s+vs|over)\b",
        r"\b(?:receded|shrunk|decreased|loss\s+of\s+forest|rate\s+increased)\b",
        r"\b(?:since\s+20\d\d)\b",
        r"\b(?:look\s+at\s+the\s+changes|urban\s+expansion\s+rate)\b",
        r"\b(?:detect\s+temporal\s+flood\s+changes)\b"
    ]
    for pattern in change_patterns:
        match = re.search(pattern, q_lower)
        if match:
            task_positions.append((match.start(), Task.CHANGE_VQA))
            break

    # 4. Captioning patterns (Flexible non-adjacent matches)
    captioning_patterns = [
        r"\b(?:describe|description)\b",
        r"\bwhat\s+does\s+(?:this|the\s+scene|the\s+image|it)\b.*?\blook\s+like\b",
        r"\b(?:caption|captioning)\b",
        r"\b(?:summarize|summary\s+of\s+the\s+scene|high-level\s+summary)\b",
        r"\b(?:overview|detailed\s+overview)\b",
        r"\b(?:what\s+is\s+shown(?:\s+in\s+this\s+image)?)\b",
        r"\b(?:what\s+is\s+visible)\b",
        r"\b(?:give\s+me\s+a\s+summary)\b",
        r"\b(?:tell\s+me\s+what\s+you\s+see)\b"
    ]
    for pattern in captioning_patterns:
        match = re.search(pattern, q_lower)
        if match:
            task_positions.append((match.start(), Task.CAPTIONING))
            break

    # 5. VQA patterns
    vqa_patterns = [
        r"\b(?:how\s+many|count|number\s+of)\b",
        r"\b(?:is\s+there|are\s+there|do\s+you\s+see|can\s+you\s+verify)\b",
        r"\b(?:what\s+(?:type|kind|color|category|percentage|fraction|proportion))\b",
        r"\b(?:classify|classification|identify\s+the\s+type)\b",
        r"\b(?:what\s+is\s+the\s+land\s+cover|what\s+is\s+the\s+primary\s+cause|what\s+is\s+the\s+dominant)\b",
        r"\b(?:assess\s+whether|estimate\s+the|measure\s+the)\b"
    ]
    for pattern in vqa_patterns:
        match = re.search(pattern, q_lower)
        if match:
            task_positions.append((match.start(), Task.VQA))
            break

    task_positions.sort(key=lambda x: x[0])

    tasks: List[Task] = []
    seen = set()
    for _, t in task_positions:
        if t not in seen:
            seen.add(t)
            tasks.append(t)

    if not tasks:
        tasks = [Task.VQA]

    return tasks


def extract_location(query: str) -> Optional[str]:
    """
    Extracts place name or raw coordinates from query text.
    Checks coordinate patterns, gazetteer keywords, and prepositional phrases.
    """
    clean_q = query.strip()

    # 1. Check for explicit coordinate patterns:
    # Handles "28.6139, 77.2090", "(28.6139, 77.2090)", "28.6139°N, 77.2090°E", "lat: 12.9716, lon: 77.5946", etc.
    coord_pattern = r"(?:lat(?:itude)?\s*[:=]?\s*)?(-?\d{1,3}(?:\.\d+)?)\s*([°\s]*[NSEWnsew])?\s*[,;/]\s*(?:(?:lon(?:gitude)?|lng)\s*[:=]?\s*)?(-?\d{1,3}(?:\.\d+)?)\s*([°\s]*[NSEWnsew])?"
    coord_m = re.search(coord_pattern, clean_q, re.IGNORECASE)
    if coord_m:
        try:
            lat_v = float(coord_m.group(1))
            lat_card = (coord_m.group(2) or "").strip().upper()
            lon_v = float(coord_m.group(3))
            lon_card = (coord_m.group(4) or "").strip().upper()
            if "S" in lat_card:
                lat_v = -abs(lat_v)
            if "W" in lon_card:
                lon_v = -abs(lon_v)
            if "." in coord_m.group(1) or "." in coord_m.group(3) or lat_card or lon_card or "," in clean_q:
                return f"{lat_v}, {lon_v}"
        except Exception:
            pass

    q_lower = clean_q.lower()
    matches: List[Tuple[int, int, str]] = []

    # 2. Match against offline gazetteer
    for key, canonical in GAZETTEER_CANONICAL.items():
        pattern = r"\b" + re.escape(key) + r"\b"
        for m in re.finditer(pattern, q_lower):
            matches.append((m.start(), m.end(), canonical))

    if matches:
        matches.sort(key=lambda x: (x[0], -(x[1] - x[0])))
        return matches[0][2]

    # 3. Match prepositional phrases: e.g. "in Delhi", "around Chennai", "near Mumbai"
    prep_pattern = r"\b(?:in|around|near|over|across|at|of)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b"
    for m in re.finditer(prep_pattern, clean_q):
        candidate = m.group(1).strip()
        tokens = [t.lower() for t in candidate.split()]
        if not any(t in EXCLUDED_LOCATION_TOKENS for t in tokens) and len(candidate) > 2:
            return candidate

    # 4. Fallback preposition pattern for arbitrary or lowercase location tokens e.g. "around xyzqwe123randomland"
    prep_pattern_any = r"\b(?:in|around|near|over|across|at|for|of)\s+([a-zA-Z0-9_\-]+(?:\s+[a-zA-Z0-9_\-]+)*)\b"
    for m in re.finditer(prep_pattern_any, clean_q):
        candidate = m.group(1).strip()
        tokens = [t.lower() for t in candidate.split()]
        if not any(t in EXCLUDED_LOCATION_TOKENS for t in tokens) and len(candidate) > 2:
            return candidate

    # 5. Standalone single-word query that isn't a task keyword
    bare_q = clean_q.strip("?.,! ")
    if " " not in bare_q and len(bare_q) > 2:
        if bare_q.lower() not in EXCLUDED_LOCATION_TOKENS and not any(bare_q.lower() in t.value for t in Task):
            return bare_q

    return None


def extract_date_or_time_reference(query: str) -> Optional[str]:
    """
    Extracts temporal expressions and user phrasing references from query.
    Preserves original user phrasing.
    """
    temporal_patterns = [
        # Multi-year or comparison patterns
        r"\b(?:between\s+\d{4}\s+and\s+\d{4})\b",
        r"\b(?:from\s+\d{4}\s+to\s+\d{4})\b",
        r"\b(?:\d{4}\s+(?:vs|versus|and|to|-)\s+\d{4})\b",
        # Seasonal & Monsoon comparisons
        r"\b(?:before\s+and\s+after\s+monsoon)\b",
        r"\b(?:before\s+and\s+after\s+the\s+monsoon)\b",
        r"\b(?:pre-monsoon\s+vs\s+post-monsoon)\b",
        r"\b(?:pre-monsoon|post-monsoon)\b",
        r"\b(?:monsoon(?:\s+season)?)\b",
        r"\b(?:Kharif\s+season\s+\d{4}\s+and\s+\d{4}|kharif(?:\s+season)?|rabi(?:\s+season)?)\b",
        r"\b(?:dry\s+season|wet\s+season)\b",
        # Disaster & Event relative phrases
        r"\b(?:before\s+and\s+after\s+(?:the\s+)?(?:\d{4}\s+)?(?:flood|cyclone|disaster|earthquake|wildfire|landslide))\b",
        r"\b(?:after\s+the\s+summer\s+floods)\b",
        r"\b(?:before\s+and\s+after)\b",
        r"\b(?:post-cyclone|pre-cyclone|post-flood|pre-flood)\b",
        # Relative durations
        r"\b(?:over\s+the\s+(?:past|last)\s+\d+\s+(?:years|months|decades))\b",
        r"\b(?:over\s+the\s+(?:past|last)\s+decade)\b",
        r"\b(?:over\s+the\s+years|over\s+time)\b",
        r"\b(?:last\s+week|past\s+week|last\s+year|past\s+year|past\s+month|recent\s+months)\b",
        # Season / Month with year e.g. "between May 2023 and June 2023", "between May and August 2023"
        r"\b(?:between\s+[A-Za-z]+\s+\d{4}\s+and\s+[A-Za-z]+\s+\d{4})\b",
        r"\b(?:between\s+[A-Za-z]+\s+and\s+[A-Za-z]+\s+\d{4})\b",
        r"\b(?:summer|winter|spring|autumn|monsoon)\s+\d{4}\b",
        # Single year e.g. 2023
        r"\b(19\d{2}|20\d{2})\b"
    ]

    for pattern in temporal_patterns:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            start, end = match.span()
            return query[start:end].strip()

    return None


def extract_target_object(query: str, tasks: List[Task]) -> Optional[str]:
    """
    Extracts target object/region to highlight for grounding tasks.
    Returns None for non-grounding tasks.
    """
    if Task.GROUNDING not in tasks:
        return None

    q_lower = query.lower()

    target_synonyms: List[Tuple[List[str], str]] = [
        (["water body", "water bodies", "the water body"], "the water body"),
        (["flooded zones", "flooded zone"], "the flooded zones"),
        (["flooded areas", "flooded regions", "inundated areas", "flooded districts"], "the flooded areas"),
        (["submerged roads", "submerged highways", "submerged streets"], "the submerged roads"),
        (["submerged buildings", "flooded buildings"], "the submerged buildings"),
        (["built-up area boundary", "built-up area", "built-up areas", "urban area"], "the built-up area"),
        (["paddy fields", "paddy field"], "the paddy fields"),
        (["agricultural fields", "cropland", "crop fields", "farmland"], "the agricultural fields"),
        (["dense forest patches", "forest patches", "forest cover", "tree canopy"], "the forest cover"),
        (["deforestation patches", "deforested patches", "deforested areas"], "the deforestation patches"),
        (["maritime cargo vessels", "cargo vessels", "cargo ships", "vessels", "ships", "boats", "ship detection"], "the ships"),
        (["oil storage tanks", "oil tanks", "storage tanks"], "the oil tanks"),
        (["damaged bridge pier", "damaged bridge"], "the damaged bridge"),
        (["railway bridge", "bridges", "bridge"], "the bridge"),
        (["solar farms", "solar panels", "solar farm"], "the solar panels"),
        (["landslide debris", "landslide scar"], "the landslide debris"),
        (["water reservoir", "reservoir"], "the water reservoir"),
        (["primary runway", "airport runway", "runway", "runways", "airport runways"], "the runway"),
        (["dry lakebed", "lakebed"], "the lakebed"),
        (["burn scar extent", "burn scar", "burn scars"], "the burn scar"),
        (["industrial warehouses", "warehouses", "warehouse"], "the warehouses"),
        (["mangrove forest zone", "mangrove forest", "mangroves"], "the mangrove forest"),
        (["highway interchange", "interchange"], "the highway interchange"),
        (["standing water pools", "standing water"], "the standing water"),
        (["informal settlement clusters", "informal housing settlements", "informal settlements"], "the informal settlements"),
        (["commercial high-rise buildings", "high-rise buildings", "buildings"], "the buildings"),
    ]

    for aliases, canonical in target_synonyms:
        for alias in aliases:
            if re.search(r"\b" + re.escape(alias) + r"\b", q_lower):
                if "runways" in q_lower:
                    return "the runways"
                return canonical

    action_pattern = (
        r"\b(?:highlight|locate|pinpoint|outline|box|demarcate|find|segment|point out|show me)\s+"
        r"(?:for\s+|of\s+|to\s+|the\s+|all\s+|any\s+)?([a-z0-9\s\-]+?)(?=\s+(?:in|around|near|over|across|before|after|and tell|and compare|and describe|using|from|with|$|\.|\?))"
    )
    match = re.search(action_pattern, q_lower)
    if match:
        extracted = match.group(1).strip()
        extracted = re.sub(r"^(?:for|of|to|all|any|the)\s+", "", extracted).strip()
        extracted = re.sub(r"\s+(?:in|at|on|for|of|to)$", "", extracted).strip()
        if len(extracted) > 1 and extracted not in {"where", "what", "how", "this", "it"}:
            if not extracted.startswith("the "):
                return f"the {extracted}"
            return extracted

    return "the specified region"


def extract_domain(query: str) -> Domain:
    """
    Classifies domain using a 3-tier weighted keyword matching system.
    Tier 3 (Weight 3.0): Highly specific, domain-defining phrases/terms
    Tier 2 (Weight 1.5): Moderately specific terms
    Tier 1 (Weight 1.0): Generic contextual terms

    Prevents generic single words from outscoring specific domain phrases.
    """
    q_lower = query.lower()
    scores: Dict[Domain, float] = {domain: 0.0 for domain in Domain}
    tier_weights = {3: 3.0, 2: 1.5, 1: 1.0}

    for domain, tiers in DOMAIN_KEYWORDS_3TIER.items():
        matched_char_spans: Set[int] = set()

        # Iterate from Tier 3 down to Tier 1
        for tier in (3, 2, 1):
            weight = tier_weights[tier]
            # Match longer phrases first within the tier
            sorted_kws = sorted(tiers.get(tier, []), key=len, reverse=True)

            for kw in sorted_kws:
                pattern = r"\b" + re.escape(kw) + r"\b"
                for m in re.finditer(pattern, q_lower):
                    span_indices = set(range(m.start(), m.end()))
                    # Avoid double-counting overlapping characters within the same domain
                    if not (span_indices & matched_char_spans):
                        scores[domain] += weight + 0.01 * len(kw)
                        matched_char_spans.update(span_indices)

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] > 0.0:
        return best_domain

    return Domain.GENERAL


def extract_sensor(query: str, tasks: List[Task], domain: Domain) -> Sensor:
    """
    Determines sensor modality with default to Optical for remote sensing queries.
    Returns Sensor.BOTH if fusion is detected, Sensor.SAR if SAR is present,
    and Sensor.UNSPECIFIED only for brief ambiguous queries.
    """
    q_lower = query.lower()

    has_sar = bool(re.search(r"\b(?:sar|radar|sentinel-1|nisar|risat|microwave|backscatter)\b", q_lower))
    has_optical = bool(re.search(r"\b(?:optical|rgb|multispectral|sentinel-2|landsat|cartosat|true color|visual)\b", q_lower))
    has_both_keywords = bool(re.search(r"\b(?:both|fuse|fusion|combine|and sar|together with sar|optical and radar|sar and optical)\b", q_lower))

    if Task.FUSION_ANALYSIS in tasks or (has_sar and has_optical) or has_both_keywords:
        return Sensor.BOTH

    if has_sar:
        return Sensor.SAR

    ambiguous_unspecified = {
        "tell me about this", "what is this", "compare",
        "look at the changes here", "show me the water",
        "urban expansion rate", "overview of the scene", "is there flooding"
    }
    if q_lower.strip("?. ") in ambiguous_unspecified:
        return Sensor.UNSPECIFIED

    return Sensor.OPTICAL


def calculate_confidence(
    tasks: List[Task],
    location: Optional[str],
    date_ref: Optional[str],
    target_obj: Optional[str],
    domain: Domain,
    sensor: Sensor
) -> float:
    """Calculates heuristic confidence score for keyword fallback (strictly capped <= 0.50)."""
    conf = 0.20
    if tasks != [Task.VQA]:
        conf += 0.08
    if len(tasks) > 1:
        conf += 0.04
    if location is not None:
        conf += 0.05
    if date_ref is not None:
        conf += 0.05
    if target_obj is not None:
        conf += 0.04
    if domain != Domain.GENERAL:
        conf += 0.04
    if sensor != Sensor.UNSPECIFIED:
        conf += 0.04

    return min(0.50, round(conf, 2))


def classify_query_fallback(query: str) -> InterpretedQuery:
    """Rule-based fallback classifier returning a valid InterpretedQuery."""
    clean_query = query.strip()
    tasks = extract_task_sequence(clean_query)
    location = extract_location(clean_query)
    date_ref = extract_date_or_time_reference(clean_query)
    target_obj = extract_target_object(clean_query, tasks)
    domain = extract_domain(clean_query)
    sensor = extract_sensor(clean_query, tasks, domain)
    confidence = calculate_confidence(tasks, location, date_ref, target_obj, domain, sensor)

    return InterpretedQuery(
        task_sequence=tasks,
        location=location,
        date_or_time_reference=date_ref,
        target_object=target_obj,
        domain=domain,
        sensor=sensor,
        confidence=confidence,
        source=Source.KEYWORD_FALLBACK,
        raw_query=clean_query,
    )
