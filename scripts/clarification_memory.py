"""
clarification_memory.py — persistent "remember what the user told us" store
for the query interpreter.

Design goal: when the interpreter is unsure (low confidence), the caller can
ask the user a clarifying question. Once answered, that answer is saved here
permanently (to a local JSON file), keyed by a normalized version of the
query text. Next time the same or a near-identical query comes in, the
interpreter checks here FIRST, before spending an LLM call at all -- so the
same ambiguous query never has to be clarified twice.

Matching strategy (deliberately simple, documented limitation):
  This does EXACT match on a normalized query string (lowercased, extra
  whitespace collapsed, trailing punctuation stripped). It does NOT do fuzzy
  or semantic matching -- "is there flooding" and "is there any flooding"
  are treated as different queries and would each need their own
  clarification. This is a real limitation: it only helps with queries that
  get asked verbatim (or near-verbatim) more than once, which is common for
  short/terse queries and demo scripts, but won't generalize to novel
  rephrasings. Upgrading to fuzzy matching (e.g. embedding similarity) is a
  reasonable future improvement but adds a real dependency and complexity;
  starting simple and exact is the right first step.
"""

import json
import re
from pathlib import Path
from typing import Optional

try:
    from .schema import InterpretedQuery
except ImportError:
    from schema import InterpretedQuery


DEFAULT_MEMORY_PATH = Path(__file__).parent / "clarification_memory.json"


def normalize_query(query: str) -> str:
    """Lowercase, collapse whitespace, strip trailing punctuation. This is
    the key every lookup/save goes through -- keep it deterministic."""
    q = query.strip().lower()
    q = re.sub(r"\s+", " ", q)
    q = re.sub(r"[?!.]+$", "", q)
    return q


class ClarificationMemory:
    """
    Thin wrapper around a JSON file mapping normalized_query -> stored
    InterpretedQuery (as a dict). Loads on construction, writes through on
    every remember() call (no explicit save() needed, so a crash mid-session
    doesn't lose earlier clarifications).
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or DEFAULT_MEMORY_PATH
        self._store: dict = {}
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    self._store = json.load(f)
            except (json.JSONDecodeError, OSError):
                # Corrupt or unreadable file -- start fresh rather than crash.
                # This trades "lose some remembered clarifications" for
                # "never block the pipeline on a bad cache file", which is
                # the right tradeoff for a convenience cache like this.
                self._store = {}
        else:
            self._store = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._store, f, indent=2)

    def lookup(self, query: str) -> Optional[InterpretedQuery]:
        """Returns the remembered InterpretedQuery for this query, or None
        if it's never been clarified before."""
        key = normalize_query(query)
        raw = self._store.get(key)
        if raw is None:
            return None
        # raw_query in storage is the ORIGINAL clarified query, not this
        # call's exact text -- update it so the returned object reflects
        # what was actually asked this time.
        raw = dict(raw)
        raw["raw_query"] = query.strip()
        return InterpretedQuery.model_validate(raw)

    def remember(self, query: str, interpreted: InterpretedQuery) -> None:
        """Saves a (typically user-clarified) result, keyed by normalized
        query text. Overwrites any previous entry for the same key."""
        key = normalize_query(query)
        self._store[key] = interpreted.model_dump(mode="json")
        self._save()

    def forget(self, query: str) -> bool:
        """Removes a remembered clarification, e.g. if it turns out to be
        wrong. Returns True if something was actually removed."""
        key = normalize_query(query)
        if key in self._store:
            del self._store[key]
            self._save()
            return True
        return False

    def __len__(self) -> int:
        return len(self._store)
