"""
model_registry.py — Specialist Model Registry & Loader Functions for SatQuery AI.

Registers all specialist models and the resident interpreter LLM with the
ModelLifecycleManager. Ensures accurate class imports, constructors, and
unloading hooks. Supports both standard specialist names and router aliases.
"""

from typing import Any, Callable, Optional
import os
import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from .model_lifecycle_manager import ModelLifecycleManager, ModelSpec, Residency
except ImportError:
    from model_lifecycle_manager import ModelLifecycleManager, ModelSpec, Residency

# Specialist names as registered with the lifecycle manager.
GENERAL_VLM = "general_vlm"
GEOCHAT = "geochat"
QWEN_CDVQA = "qwen_cdvqa"
QWEN_CHANGE = "qwen_change"  # Router alias for change_vqa
FUSION = "fusion"
FUSION_MODEL = "fusion_model"  # Router alias for fusion_analysis
INTERPRETER_LLM = "interpreter_llm"

# Task-type (from interpreter.py's InterpretedQuery.task_sequence) -> which
# specialist actually handles it.
TASK_TO_SPECIALIST = {
    "vqa": GENERAL_VLM,
    "captioning": GENERAL_VLM,
    "grounding": GENERAL_VLM,
    "geochat": GEOCHAT,
    "change_vqa": QWEN_CDVQA,
    "fusion_analysis": FUSION,
}


class InterpreterLLMWrapper:
    """
    Wrapper for Qwen3.5-2B (or configurable HuggingFace CausalLM) providing a
    Callable[[str], str] interface matching QueryInterpreter's llm_fn contract.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3.5-2B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.model_name = os.environ.get("INTERPRETER_MODEL_NAME", model_name)
        self.device = device
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        """Loads the tokenizer and 4-bit quantized model (or CPU float32)."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if torch.cuda.is_available() and self.device.startswith("cuda"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
            )
        self.model.eval()

    def __call__(self, prompt: str) -> str:
        """
        Callable[[str], str] interface matching QueryInterpreter's llm_fn.
        Takes prompt string, runs generation, and decodes raw completion.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("InterpreterLLM is not loaded. Call load() first.")

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=False,
            )

        decoded = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return decoded

    def unload(self) -> None:
        """Releases GPU memory and deletes model/tokenizer references."""
        del self.model
        del self.tokenizer
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _load_qwen_cdvqa():
    """
    Loads ChangeVQASpecialist from qwen_specialist.py.
    Note: Default constructor uses paths in qwen_specialist.py:
      BASE_MODEL_PATH and ADAPTER_PATH.
    """
    try:
        from .qwen_specialist import ChangeVQASpecialist
    except ImportError:
        from qwen_specialist import ChangeVQASpecialist
    specialist = ChangeVQASpecialist()
    specialist.load()
    return specialist


def _load_general_vlm():
    """
    Loads GeneralVLMSpecialist from general_vlm_specialist.py.
    Shares the underlying Qwen2.5-VL-3B-Instruct model + LoRA with ChangeVQASpecialist.
    """
    try:
        from .general_vlm_specialist import GeneralVLMSpecialist
    except ImportError:
        from general_vlm_specialist import GeneralVLMSpecialist
    specialist = GeneralVLMSpecialist()
    specialist.load()
    return specialist


def _load_geochat():
    """
    Loads GeoChatSpecialist for GeoChat VLM inference.
    """
    try:
        from .geochat_specialist import GeoChatSpecialist
    except ImportError:
        from geochat_specialist import GeoChatSpecialist
    specialist = GeoChatSpecialist()
    specialist.load()
    return specialist


def _load_fusion():
    """
    Loads FusionSpecialist from fusion_model.py.
    Note: Default constructor uses checkpoint_path in fusion_model.py.
    """
    try:
        from .fusion_model import FusionSpecialist
    except ImportError:
        from fusion_model import FusionSpecialist
    specialist = FusionSpecialist()
    specialist.load()
    return specialist


def _load_interpreter_llm():
    """
    Loads Qwen3.5-2B wrapped as an InterpreterLLMWrapper satisfying
    Callable[[str], str] for QueryInterpreter.
    """
    llm = InterpreterLLMWrapper()
    llm.load()
    return llm


def _unload_specialist(specialist) -> None:
    """
    Invokes the specialist's .unload() method when evicted from the swap slot.
    Passes force_release=True to release shared GPU singletons if evicting the slot.
    """
    if hasattr(specialist, "unload") and callable(specialist.unload):
        try:
            specialist.unload(force_release=True)
        except TypeError:
            specialist.unload()


def build_registry(device: str = "cuda") -> ModelLifecycleManager:
    """Creates and fully populates the lifecycle manager. Call this once
    at app startup."""
    manager = ModelLifecycleManager(device=device)

    # Resident interpreter LLM (stays resident alongside Qwen2.5-VL-3B: 1.45GB + 2.42GB = 3.87GB < 5.64GB)
    manager.register(ModelSpec(
        name=INTERPRETER_LLM,
        loader_fn=_load_interpreter_llm,
        residency=Residency.RESIDENT,
        unloader_fn=_unload_specialist,
    ))

    # General VLM Specialist (single-instance base Qwen2.5-VL with LoRA disabled)
    manager.register(ModelSpec(
        name=GENERAL_VLM,
        loader_fn=_load_general_vlm,
        residency=Residency.SWAPPABLE,
        unloader_fn=_unload_specialist,
        shared_group="qwen_vl",
    ))

    # GeoChat Specialist (isolated swap slot, not in qwen_vl shared group)
    manager.register(ModelSpec(
        name=GEOCHAT,
        loader_fn=_load_geochat,
        residency=Residency.SWAPPABLE,
        unloader_fn=_unload_specialist,
    ))

    # Change VQA (registered under standard name and router alias) — shares "qwen_vl" slot group
    manager.register(ModelSpec(
        name=QWEN_CDVQA,
        loader_fn=_load_qwen_cdvqa,
        residency=Residency.SWAPPABLE,
        unloader_fn=_unload_specialist,
        shared_group="qwen_vl",
    ))
    manager.register(ModelSpec(
        name=QWEN_CHANGE,
        loader_fn=_load_qwen_cdvqa,
        residency=Residency.SWAPPABLE,
        unloader_fn=_unload_specialist,
        shared_group="qwen_vl",
    ))

    # Optical-SAR Fusion (registered under standard name and router alias)
    manager.register(ModelSpec(
        name=FUSION,
        loader_fn=_load_fusion,
        residency=Residency.SWAPPABLE,
        unloader_fn=_unload_specialist,
    ))
    manager.register(ModelSpec(
        name=FUSION_MODEL,
        loader_fn=_load_fusion,
        residency=Residency.SWAPPABLE,
        unloader_fn=_unload_specialist,
    ))

    return manager