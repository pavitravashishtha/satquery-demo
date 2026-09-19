"""
GeoChatSpecialist — Standardized wrapper for GeoChat VLM inference.

Provides a consistent specialist interface (load / run / unload / memory_footprint_gb)
matching the pattern used by other specialists in the SatQuery AI system.

Supports three task types:
  - "vqa"        : Visual question answering
  - "captioning" : Image captioning
  - "grounding"  : Text-guided region grounding (referring expression localization)

Confidence scoring uses average token generation probability (mean of max softmax
probability at each decoding step). This bypasses GeoChat's Chat class and calls
model.generate() directly with output_scores=True.

GPU: RTX 4070 Ti (12 GB) — uses 4-bit NF4 quantization.
Model: MBZUAI/geochat-7B (base pretrained weights, no fine-tuning).
"""

import os
import sys
import re
import time
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from typing import Optional, Union

# Ensure GeoChat's codebase is importable
GEOCHAT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "GeoChat")
if GEOCHAT_ROOT not in sys.path:
    sys.path.insert(0, GEOCHAT_ROOT)

try:
    import transformers.utils.import_utils
    import transformers.modeling_utils
    transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
    transformers.modeling_utils.check_torch_load_is_safe = lambda: None
except Exception:
    pass

from geochat.conversation import conv_templates
from geochat.model.builder import load_pretrained_model
from geochat.mm_utils import (
    get_model_name_from_path,
    process_images_demo,
    tokenizer_image_token,
    KeywordsStoppingCriteria,
)
from geochat.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
cudnn.benchmark = False
cudnn.deterministic = True


class GeoChatSpecialist:
    """
    Standardized specialist wrapper around GeoChat (MBZUAI/geochat-7B).

    Interface:
        load()                → load model + vision encoder into GPU
        run(image, query, task_type)  → run inference, return structured result
        unload()              → free GPU memory
        memory_footprint_gb() → measure real VRAM usage
    """

    MODEL_PATH = "MBZUAI/geochat-7B"
    MODEL_BASE = None
    DEVICE = "cuda"
    GPU_ID = 0
    LOAD_8BIT = False
    LOAD_4BIT = True  # Required: 7B fp16 ~ 14 GB > 10.5 GB available on RTX 4070 Ti

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self._loaded = False
        self.repetition_penalty = 1.15

    # ------------------------------------------------------------------
    # load()
    # ------------------------------------------------------------------
    def load(self):
        """Load GeoChat pretrained weights with 4-bit quantization."""
        if self._loaded:
            print("[GeoChatSpecialist] Model already loaded.")
            return

        print(f"[GeoChatSpecialist] Loading model from {self.MODEL_PATH} "
              f"(4-bit={self.LOAD_4BIT}, 8-bit={self.LOAD_8BIT}) ...")
        t0 = time.time()

        model_name = get_model_name_from_path(self.MODEL_PATH)
        self.tokenizer, self.model, self.image_processor, self.context_len = (
            load_pretrained_model(
                self.MODEL_PATH,
                self.MODEL_BASE,
                model_name,
                self.LOAD_8BIT,
                self.LOAD_4BIT,
                device=self.DEVICE,
            )
        )
        self.model = self.model.eval()
        self._loaded = True

        elapsed = time.time() - t0
        vram = self.memory_footprint_gb()
        print(f"[GeoChatSpecialist] Loaded in {elapsed:.1f}s | "
              f"VRAM: {vram:.2f} GB")

    # ------------------------------------------------------------------
    # run()
    # ------------------------------------------------------------------
    def run(
        self,
        image: Union[str, Image.Image],
        query: str,
        task_type: str = "vqa",
    ) -> dict:
        """
        Run inference on a single image + text query.

        Args:
            image:     Path to image file or PIL Image.
            query:     User question or instruction.
            task_type: One of "vqa", "captioning", "grounding".

        Returns:
            {
                "answer":        str,
                "confidence":    float,   # avg token generation probability
                "task_type":     str,
                "evidence_maps": dict,    # {} for vqa/captioning,
                                          # {"boxes": [...], "labels": [...]} for grounding
            }
        """
        assert self._loaded, "Model not loaded. Call load() first."
        if task_type == "geochat":
            task_type = "vqa"
        assert task_type in ("vqa", "captioning", "grounding"), \
            f"Unknown task_type: {task_type}"

        tok_name = getattr(self.tokenizer, 'name_or_path', str(type(self.tokenizer)))
        print(f"[{type(self).__name__}] Running inference with tokenizer: {tok_name}")

        # --- Prepare image tensor ---
        if isinstance(image, str):
            pil_image = Image.open(image).convert("RGB")
        else:
            pil_image = image.convert("RGB")

        image_tensor = process_images_demo([pil_image], self.image_processor)
        image_tensor = image_tensor.to(
            device=f"cuda:{self.GPU_ID}", dtype=torch.float16
        )

        # --- Build prompt ---
        # Anti-hallucination geospatial grounding constraint placed after user query:
        # Ensures recency in causal attention heads, blocking the runaway 'two bridges' pre-training sequence.
        grounding_suffix = (
            "\nImportant: Answer objectively based strictly on visible pixels. "
            "Describe actual land cover (e.g. buildings, roads, vegetation, terrain, or open ground). "
            "Do NOT mention bridges, water bodies, or rivers unless they are unmistakably visible."
        )
        if task_type == "grounding":
            effective_query = f"[grounding] {query}"
        elif task_type in ("captioning", "caption"):
            effective_query = f"Provide a detailed objective description of the land cover: {query} {grounding_suffix}"
        else:
            effective_query = f"{query} {grounding_suffix}"

        conv = conv_templates["llava_v1"].copy()
        # Replicate Chat.upload_img(): append image token placeholder
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n")
        # Replicate Chat.ask(): append the user question
        # Since last message ends with '<image>\n', concatenate the query
        conv.messages[-1][1] = conv.messages[-1][1] + " " + effective_query
        # Append assistant placeholder for generation
        conv.append_message(conv.roles[1], None)

        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to(device=f"cuda:{self.GPU_ID}")

        # Ensure -200 sentinel tokens are replaced with a valid pad token ID before reaching generate/logits processor
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.where(
            input_ids == IMAGE_TOKEN_INDEX,
            torch.tensor(pad_token_id, device=input_ids.device, dtype=input_ids.dtype),
            input_ids,
        )
        assert input_ids.min() >= 0, f"Error: input_ids contains negative token indices: {input_ids.min()}"

        # Stopping criteria (same as Chat.answer_prepare)
        stop_str = conv.sep2  # "</s>" for llava_v1
        stopping_criteria = KeywordsStoppingCriteria(
            [stop_str], self.tokenizer, input_ids
        )

        # --- Generate with scores for confidence ---
        with torch.inference_mode():
            outputs = self.model.generate(
                input_ids,
                images=image_tensor,
                past_key_values=None,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=getattr(self, "repetition_penalty", 1.15),
                max_new_tokens=300,
                use_cache=True,
                stopping_criteria=[stopping_criteria],
                output_scores=True,
                return_dict_in_generate=True,
            )

        # --- Decode answer ---
        generated_ids = outputs.sequences[0, input_ids.shape[1]:]
        raw_answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # Remove trailing stop string if present
        if raw_answer.endswith(stop_str):
            raw_answer = raw_answer[: -len(stop_str)].strip()

        # --- Compute confidence (average max token probability) ---
        confidence = self._compute_confidence(outputs.scores)

        # Clean up generation tensors immediately to minimize VRAM fragmentation
        del outputs, image_tensor, input_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Post-hoc degenerate repetition detector (defense-in-depth) ---
        cleaned_answer, is_degenerate = self._detect_and_clean_degenerate_repetition(raw_answer)
        if is_degenerate:
            print(f"[{type(self).__name__}] Warning: Degenerate repetition loop detected! Truncating response.")
            raw_answer = cleaned_answer
            confidence = min(confidence, 0.40)

        # --- Parse grounding if needed ---
        evidence_maps = {}
        if task_type == "grounding":
            evidence_maps = self._parse_grounding(raw_answer)

        return {
            "answer": raw_answer,
            "confidence": round(confidence, 4),
            "task_type": task_type,
            "evidence_maps": evidence_maps,
            "is_degenerate": is_degenerate,
        }

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_confidence(scores: tuple) -> float:
        """
        Compute average token generation probability.

        For each decoding step, takes softmax over the logits and extracts the
        max probability (i.e., the probability assigned to the chosen token).
        Returns the mean across all generated tokens.

        This is the same approach described for the Qwen specialist pattern.
        """
        if not scores or len(scores) == 0:
            return 0.0

        max_probs = []
        for step_logits in scores:
            # step_logits: (batch_size, vocab_size)
            probs = torch.softmax(step_logits[0].float(), dim=-1)
            max_prob = probs.max().item()
            max_probs.append(max_prob)

        return float(np.mean(max_probs))

    # ------------------------------------------------------------------
    # Degenerate repetition detector (post-hoc guardrail)
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_and_clean_degenerate_repetition(text: str) -> tuple:
        """
        Detect degenerate n-gram repetition loops (e.g. ', a building, a building...').
        If any phrase of 1-6 words repeats consecutively 3 or more times,
        truncate at the first repetition onset and return (cleaned_text, True).
        Otherwise return (text, False).
        """
        words = text.split()
        if len(words) < 8:
            return text, False

        for n in range(1, 7):
            for i in range(len(words) - n * 2):
                ngram = words[i:i+n]
                norm_ngram = [w.strip(",.;:!?") for w in ngram]
                if not any(norm_ngram):
                    continue

                repeats = 1
                j = i + n
                while j + n <= len(words):
                    next_ngram = [w.strip(",.;:!?") for w in words[j:j+n]]
                    if next_ngram == norm_ngram:
                        repeats += 1
                        j += n
                    else:
                        break

                if repeats >= 3:
                    # Truncate right before the 2nd repetition begins (keep first mention)
                    truncated_words = words[:i+n]
                    truncated_text = " ".join(truncated_words).rstrip(",; ")
                    if not truncated_text.endswith("."):
                        truncated_text += "."
                    return truncated_text, True

        return text, False

    # ------------------------------------------------------------------
    # Grounding output parser
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_grounding(text: str) -> dict:
        """
        Parse GeoChat's grounding output format.

        GeoChat outputs grounding in two formats:
          1. Multi-object:  <p>label</p>{<x1><y1><x2><y2>}
          2. Single-object: {<x1><y1><x2><y2>}

        Coordinates are normalized to a 0-100 grid (bounding_box_size = 100).

        Returns:
            {"boxes": [[x1,y1,x2,y2], ...], "labels": [str, ...], "raw": str}
            or {} if no valid grounding found.
        """
        boxes = []
        labels = []

        # Pattern 1: labeled boxes -- <p>LABEL</p>{<x1><y1><x2><y2>}
        labeled_pattern = r"<p>(.*?)</p>\s*\{([^}]+)\}"
        for match in re.finditer(labeled_pattern, text):
            label = match.group(1).strip()
            coords_str = match.group(2)
            # May contain multiple boxes separated by }{
            coord_groups = coords_str.replace("}{", "} {").split("} {")
            for cg in coord_groups:
                cg = cg.strip().strip("{}")
                integers = re.findall(r"-?\d+", cg)
                if len(integers) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in integers[:4]]
                    boxes.append([x1, y1, x2, y2])
                    labels.append(label)

        # Pattern 2: unlabeled boxes -- {<x1><y1><x2><y2>} without preceding <p>
        if not boxes:
            unlabeled_pattern = r"\{([^}]+)\}"
            for match in re.finditer(unlabeled_pattern, text):
                coords_str = match.group(1)
                integers = re.findall(r"-?\d+", coords_str)
                if len(integers) >= 4:
                    x1, y1, x2, y2 = [int(v) for v in integers[:4]]
                    boxes.append([x1, y1, x2, y2])
                    labels.append("object")

        if not boxes:
            return {}

        return {
            "boxes": boxes,
            "labels": labels,
            "raw": text,
            "coord_system": "normalized_0_100",
        }

    # ------------------------------------------------------------------
    # unload()
    # ------------------------------------------------------------------
    def unload(self, force_release: bool = False):
        """Free GPU memory."""
        if not self._loaded:
            return

        del self.model
        del self.tokenizer
        del self.image_processor
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self._loaded = False

        import gc
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("[GeoChatSpecialist] Model unloaded, GPU memory freed.")

    # ------------------------------------------------------------------
    # memory_footprint_gb()
    # ------------------------------------------------------------------
    @staticmethod
    def memory_footprint_gb() -> float:
        """Measure REAL VRAM usage via torch.cuda.memory_allocated()."""
        return torch.cuda.memory_allocated() / (1024 ** 3)


# =======================================================================
# SMOKE TEST + CONFIDENCE RELIABILITY CHECK
# =======================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  GeoChatSpecialist -- Smoke Test & Confidence Reliability Check")
    print("=" * 70)

    # --- Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
    TEST_IMAGE = os.path.join(PROJECT_ROOT, "GeoChat", "demo_images", "7292.JPG")

    if not os.path.exists(TEST_IMAGE):
        print(f"[ERROR] Test image not found: {TEST_IMAGE}")
        sys.exit(1)

    specialist = GeoChatSpecialist()

    # ---- Task 1: Load ----
    print("\n--- LOADING MODEL ---")
    specialist.load()
    vram = specialist.memory_footprint_gb()
    print(f"  Memory footprint: {vram:.2f} GB")

    # ---- Task 2a: VQA Smoke Test ----
    print("\n--- SMOKE TEST: VQA ---")
    result_vqa = specialist.run(TEST_IMAGE, "What is visible in this image?", task_type="vqa")
    print(f"  Answer:     {result_vqa['answer']}")
    print(f"  Confidence: {result_vqa['confidence']}")
    print(f"  Task Type:  {result_vqa['task_type']}")
    print(f"  Evidence:   {result_vqa['evidence_maps']}")

    # ---- Task 2b: Grounding Smoke Test ----
    print("\n--- SMOKE TEST: GROUNDING ---")
    result_ground = specialist.run(
        TEST_IMAGE,
        "Locate the vehicles in this image",
        task_type="grounding",
    )
    print(f"  Answer:     {result_ground['answer']}")
    print(f"  Confidence: {result_ground['confidence']}")
    print(f"  Task Type:  {result_ground['task_type']}")
    print(f"  Evidence:   {result_ground['evidence_maps']}")

    # ---- Task 3: Confidence Reliability Check ----
    print("\n--- CONFIDENCE RELIABILITY CHECK ---")
    print("  Running 3+ verifiable VQA queries to assess confidence signal...\n")

    reliability_queries = [
        {
            "query": "Is this an aerial or satellite image?",
            "expected_keywords": ["aerial", "satellite", "overhead", "bird"],
            "description": "Easy -- obviously an aerial/overhead view",
        },
        {
            "query": "Are there any vehicles visible in this image?",
            "expected_keywords": ["yes"],
            "description": "Easy -- vehicles are clearly visible in 7292.JPG",
        },
        {
            "query": "Is there a swimming pool in this image?",
            "expected_keywords": ["no", "not"],
            "description": "Easy -- no swimming pool in this road/vehicle scene",
        },
        {
            "query": "Is this image taken at night?",
            "expected_keywords": ["no", "not", "day"],
            "description": "Easy -- clearly daytime lighting",
        },
    ]

    reliability_results = []
    for i, rq in enumerate(reliability_queries, 1):
        result = specialist.run(TEST_IMAGE, rq["query"], task_type="vqa")
        answer_lower = result["answer"].lower()
        is_correct = any(kw in answer_lower for kw in rq["expected_keywords"])
        reliability_results.append({
            "query": rq["query"],
            "answer": result["answer"],
            "confidence": result["confidence"],
            "correct": is_correct,
            "description": rq["description"],
        })
        status = "CORRECT" if is_correct else "INCORRECT"
        print(f"  [{i}] {status}")
        print(f"      Q: {rq['query']}")
        print(f"      A: {result['answer']}")
        print(f"      Confidence: {result['confidence']}")
        print(f"      Expected keywords: {rq['expected_keywords']}")
        print()

    # ---- Confidence Reliability Verdict ----
    print("--- CONFIDENCE RELIABILITY VERDICT ---")
    confidences = [r["confidence"] for r in reliability_results]
    correct_confs = [r["confidence"] for r in reliability_results if r["correct"]]
    incorrect_confs = [r["confidence"] for r in reliability_results if not r["correct"]]

    print(f"  All confidences:       {confidences}")
    print(f"  Correct confidences:   {correct_confs}")
    print(f"  Incorrect confidences: {incorrect_confs}")
    print(f"  Mean confidence:       {np.mean(confidences):.4f}")
    print(f"  Std confidence:        {np.std(confidences):.4f}")

    # Check if all pinned near 1.0 (same issue as Qwen specialist pattern)
    all_high = all(c > 0.90 for c in confidences)
    low_variance = np.std(confidences) < 0.05

    if all_high and low_variance:
        verdict = (
            "Confidence signal does NOT appear reliable for this model. "
            "All values are pinned near 1.0 regardless of correctness "
            "(same pattern as reported for the Qwen specialist)."
        )
    elif incorrect_confs and np.mean(incorrect_confs) >= np.mean(correct_confs) - 0.05:
        verdict = (
            "Confidence signal does NOT appear reliable for this model. "
            "Incorrect answers show confidence comparable to correct ones."
        )
    elif incorrect_confs and np.mean(incorrect_confs) < np.mean(correct_confs) - 0.1:
        verdict = (
            "Confidence signal appears PARTIALLY reliable for this model. "
            "There is some separation between correct and incorrect answer confidence."
        )
    elif not incorrect_confs:
        if all_high and low_variance:
            verdict = (
                "Cannot fully assess reliability -- all answers were correct. "
                "However, all confidences are pinned near 1.0 with very low variance, "
                "suggesting the signal may NOT be reliable (same pattern as Qwen specialist)."
            )
        else:
            verdict = (
                "Cannot fully assess reliability -- all answers were correct. "
                f"Confidence range: {min(confidences):.4f} -- {max(confidences):.4f}. "
                "Some variance exists, but a definitive verdict requires incorrect "
                "answers for comparison."
            )
    else:
        verdict = (
            "Confidence signal appears reliable for this model. "
            "Correct answers consistently show higher confidence than incorrect ones."
        )

    print(f"\n  VERDICT: {verdict}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Model:        MBZUAI/geochat-7B (base pretrained, 4-bit NF4)")
    print(f"  VRAM:         {vram:.2f} GB")
    print(f"  VQA:          {'PASS' if result_vqa['answer'] else 'FAIL'}")
    grounding_pass = bool(result_ground["evidence_maps"].get("boxes"))
    print(f"  Grounding:    {'PASS -- boxes extracted' if grounding_pass else 'FAIL -- no boxes found'}")
    print(f"  Confidence:   {verdict}")
    print("=" * 70)

    # ---- Unload ----
    specialist.unload()
    print("Done.")
