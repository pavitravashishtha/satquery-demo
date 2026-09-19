"""
model_lifecycle_manager.py — Load/unload orchestration for SatQuery AI specialists.

Manages which models are resident on the GPU at any given time, under a
strict VRAM budget (6GB target). Two categories of model:

  - RESIDENT models: small enough to stay loaded permanently (e.g. the
    Qwen3.5-2B query interpreter). Never unloaded once loaded.
  - SWAPPABLE models: heavier specialists (GeoChat, the fusion model) that
    share a single "swap slot" — only one swappable model is on GPU at a
    time. Loading a different swappable model unloads whichever one
    currently occupies the slot first.

This module deliberately does NOT know anything about GeoChat, Qwen, or the
fusion model specifically — it's given a `loader_fn` per model at
registration time and treats every model as an opaque object. This makes it
testable with plain dummy objects (see test_model_lifecycle_manager.py)
before ever touching real multi-GB weights, which is the point: swap
timing and memory-release behavior are orchestration bugs, not model
quality bugs, and should be caught here first.

Real GPU/VRAM calls (torch.cuda.empty_cache, etc.) only fire when torch and
CUDA are actually available — this file runs and is fully testable on a
machine with no GPU at all, which is what makes offline validation possible.
"""

import gc
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    # Broad except deliberately: a broken CUDA install, missing shared
    # libraries, or driver mismatch can raise OSError/ImportError/other
    # exceptions here depending on the machine. Any of these should fall
    # back to logic-only mode (no real VRAM calls) rather than crash the
    # whole module on import.
    TORCH_AVAILABLE = False

logger = logging.getLogger("SatQuery.ModelLifecycleManager")


def _cuda_available() -> bool:
    return TORCH_AVAILABLE and torch.cuda.is_available()


class Residency(str, Enum):
    RESIDENT = "resident"      # stays loaded forever once loaded (e.g. Qwen)
    SWAPPABLE = "swappable"    # shares the single swap slot (e.g. GeoChat, fusion model)


@dataclass
class ModelSpec:
    """
    Registration record for a model the manager can load/unload.

    loader_fn: takes no arguments, returns the loaded model object. Called
      exactly once per load — the manager caches the result until unload.
    unloader_fn: optional extra cleanup beyond `del` + cache clearing (e.g.
      if the model wraps a resource the manager doesn't know about). Most
      models won't need this.
    residency: RESIDENT or SWAPPABLE — see module docstring.
    shared_group: optional group identifier (e.g. 'qwen_vl') for models that
      share the same underlying GPU weights/instance. Models in the same
      shared group do not trigger an unload/swap when switching between them.
    """
    name: str
    loader_fn: Callable[[], Any]
    residency: Residency
    unloader_fn: Optional[Callable[[Any], None]] = None
    shared_group: Optional[str] = None


@dataclass
class LoadEvent:
    """One entry in the manager's event log — useful for diagnosing swap
    thrashing or unexpectedly slow loads during testing."""
    timestamp: float
    action: str  # "load", "unload", "skip_already_loaded", "skip_resident_coexist"
    model_name: str
    duration_seconds: float = 0.0
    vram_allocated_gb: Optional[float] = None
    vram_reserved_gb: Optional[float] = None


class ModelLifecycleManager:
    """
    Orchestrates which models are loaded on GPU, enforcing a single-swap-slot
    policy for heavy models while allowing resident models to stay loaded
    permanently. See module docstring for the resident/swappable distinction.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._specs: Dict[str, ModelSpec] = {}
        self._loaded: Dict[str, Any] = {}          # name -> loaded model object
        self._swap_slot_occupant: Optional[str] = None  # name of current swappable model, or None
        self.event_log: List[LoadEvent] = []

    # -------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------

    def register(self, spec: ModelSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Model '{spec.name}' is already registered")
        self._specs[spec.name] = spec
        logger.info(f"Registered model '{spec.name}' as {spec.residency.value}")

    # -------------------------------------------------------------------
    # VRAM introspection
    # -------------------------------------------------------------------

    def vram_snapshot(self) -> Dict[str, Optional[float]]:
        """
        Returns current allocated/reserved VRAM in GB, or None if no CUDA
        device is available (e.g. running on CPU during logic testing).
        """
        if not _cuda_available():
            return {"allocated_gb": None, "reserved_gb": None}
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
        }

    def _log_event(self, action: str, model_name: str, duration: float = 0.0) -> None:
        snap = self.vram_snapshot()
        self.event_log.append(LoadEvent(
            timestamp=time.time(),
            action=action,
            model_name=model_name,
            duration_seconds=duration,
            vram_allocated_gb=snap["allocated_gb"],
            vram_reserved_gb=snap["reserved_gb"],
        ))

    # -------------------------------------------------------------------
    # Core load / unload
    # -------------------------------------------------------------------

    def load(self, name: str) -> Any:
        """
        Ensures `name` is loaded and returns the model object. If it's a
        swappable model and a *different* swappable model currently
        occupies the slot, that one is unloaded first. If it's already
        loaded (resident or swappable), returns the cached object with no
        extra work.
        """
        if name not in self._specs:
            raise KeyError(f"Model '{name}' was never registered")

        if name in self._loaded:
            self._log_event("skip_already_loaded", name)
            return self._loaded[name]

        spec = self._specs[name]

        if spec.residency == Residency.SWAPPABLE:
            if self._swap_slot_occupant is not None and self._swap_slot_occupant != name:
                current_spec = self._specs.get(self._swap_slot_occupant)
                if (
                    spec.shared_group is not None
                    and current_spec is not None
                    and current_spec.shared_group == spec.shared_group
                ):
                    logger.info(
                        f"Model '{name}' shares slot group '{spec.shared_group}' "
                        f"with occupant '{self._swap_slot_occupant}'. Skipping swap unload."
                    )
                else:
                    self.unload(self._swap_slot_occupant)

            # Heavy 7B models (like GeoChat) cannot co-exist with resident interpreter LLM on a 6GB GPU
            if name == "geochat" and "interpreter_llm" in self._loaded:
                logger.info("Evicting resident interpreter_llm to provide full VRAM headroom for GeoChat-7B.")
                self.unload("interpreter_llm")

        if name == "interpreter_llm" and "geochat" in self._loaded:
            logger.info("Evicting geochat to allow interpreter_llm reload.")
            self.unload("geochat")

        start = time.time()
        model = spec.loader_fn()
        duration = time.time() - start

        self._loaded[name] = model
        if spec.residency == Residency.SWAPPABLE:
            self._swap_slot_occupant = name

        self._log_event("load", name, duration)
        logger.info(f"Loaded '{name}' in {duration:.2f}s")
        return model

    def unload(self, name: str) -> None:
        """
        Unloads a model and verifies memory was actually released. Safe to
        call on a model that isn't currently loaded (no-op).

        The three-step release sequence (del -> gc.collect -> empty_cache)
        is deliberate and in this order — skipping any step is the most
        common reason people think they've freed VRAM but haven't.
        """
        if name not in self._loaded:
            return

        spec = self._specs[name]
        before = self.vram_snapshot()

        model = self._loaded.pop(name)
        if spec.unloader_fn is not None:
            spec.unloader_fn(model)
        del model

        # If evicting a swappable model with a shared_group, clean up other loaded members in that group
        if spec.residency == Residency.SWAPPABLE and self._swap_slot_occupant == name:
            self._swap_slot_occupant = None
            if spec.shared_group is not None:
                co_names = [
                    m for m, s in self._specs.items()
                    if s.shared_group == spec.shared_group and m in self._loaded
                ]
                for co_name in co_names:
                    co_spec = self._specs[co_name]
                    co_mod = self._loaded.pop(co_name)
                    if co_spec.unloader_fn is not None:
                        co_spec.unloader_fn(co_mod)
                    del co_mod

        gc.collect()
        if _cuda_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        after = self.vram_snapshot()

        self._log_event("unload", name)
        logger.info(f"Unloaded '{name}'")

        if _cuda_available() and before["allocated_gb"] is not None and after["allocated_gb"] is not None:
            freed = before["allocated_gb"] - after["allocated_gb"]
            if freed <= 0:
                logger.warning(
                    f"Unloading '{name}' did not reduce allocated VRAM "
                    f"(before={before['allocated_gb']:.3f}GB, after={after['allocated_gb']:.3f}GB). "
                    f"Memory may not have actually been released — investigate before trusting the swap slot is free."
                )

    def is_loaded(self, name: str) -> bool:
        return name in self._loaded

    def is_registered(self, name: str) -> bool:
        """Returns True if a model spec with `name` has been registered."""
        return name in self._specs

    def is_available(self, name: str) -> bool:
        """Returns True if the model is registered and available for loading."""
        return name in self._specs

    def current_swap_occupant(self) -> Optional[str]:
        return self._swap_slot_occupant

    # -------------------------------------------------------------------
    # Task-sequence loop — the actual usage pattern from the controller
    # -------------------------------------------------------------------

    def run_sequence(
        self,
        task_sequence: List[str],
        dispatch_fn: Callable[[str, Any], Any],
    ) -> List[Any]:
        """
        Runs a list of tasks in order (e.g. a compound query's
        task_sequence: ["grounding", "change_vqa"]), loading whichever
        model each task needs and swapping only when the required model
        genuinely isn't already resident. Returns one result per task, in
        order.

        `dispatch_fn(task_name, model)` is supplied by the caller — this
        manager doesn't know how to actually run inference, only how to
        make sure the right model is loaded before dispatch_fn is called.

        Consecutive tasks that need the *same* model (or a resident model
        that's already up) incur no reload — this is the main reason to
        route task_sequence through here rather than loading fresh per task.
        """
        results: List[Any] = []
        for task in task_sequence:
            model = self.load(task)  # no-op if already loaded, swaps if needed
            result = dispatch_fn(task, model)
            results.append(result)
        return results
