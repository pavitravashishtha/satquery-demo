# ~/satquery/scripts/qwen_specialist.py
import os
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

# Resolved model path: check local directory first, fallback to cached HF model ID
BASE_MODEL_PATH = (
    "/home/pavitra/satquery/models/qwen2.5-vl-3b"
    if os.path.exists("/home/pavitra/satquery/models/qwen2.5-vl-3b")
    else "Qwen/Qwen2.5-VL-3B-Instruct"
)
ADAPTER_PATH = "/home/pavitra/satquery/checkpoints/qwen2.5-vl-cdvqa-lora"


# Shared singleton holder to ensure base weights + LoRA are only loaded ONCE across specialists
_SHARED_QWEN_MODEL = None
_SHARED_QWEN_PROCESSOR = None


def get_shared_qwen_vl(base_model_path=BASE_MODEL_PATH, adapter_path=ADAPTER_PATH):
    """
    Returns the loaded (model, processor) singleton. If not already loaded,
    initializes Qwen2.5-VL-3B-Instruct in 4-bit NF4 with the CDVQA LoRA adapter.
    Both ChangeVQASpecialist and GeneralVLMSpecialist share these exact weights.
    """
    global _SHARED_QWEN_MODEL, _SHARED_QWEN_PROCESSOR
    if _SHARED_QWEN_MODEL is None or _SHARED_QWEN_PROCESSOR is None:
        processor = AutoProcessor.from_pretrained(base_model_path)
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_model_path,
            quantization_config=bnb_config,
            device_map="auto",
        )
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        peft_model.eval()
        _SHARED_QWEN_MODEL = peft_model
        _SHARED_QWEN_PROCESSOR = processor
    return _SHARED_QWEN_MODEL, _SHARED_QWEN_PROCESSOR


def release_shared_qwen_vl():
    """Explicitly deletes the shared Qwen-VL instance and flushes CUDA cache."""
    global _SHARED_QWEN_MODEL, _SHARED_QWEN_PROCESSOR
    if _SHARED_QWEN_MODEL is not None:
        del _SHARED_QWEN_MODEL
        _SHARED_QWEN_MODEL = None
    if _SHARED_QWEN_PROCESSOR is not None:
        del _SHARED_QWEN_PROCESSOR
        _SHARED_QWEN_PROCESSOR = None
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class ChangeVQASpecialist:
    """SpecialistModel interface wrapper for the fine-tuned Qwen change-VQA model."""

    def __init__(self, base_model_path=BASE_MODEL_PATH, adapter_path=ADAPTER_PATH, device="cuda"):
        self.base_model_path = base_model_path
        self.adapter_path = adapter_path
        self.device = device
        self.model = None
        self.processor = None

    def load(self):
        self.model, self.processor = get_shared_qwen_vl(self.base_model_path, self.adapter_path)
        lora_applied = isinstance(self.model, PeftModel) and bool(getattr(self.model, "peft_config", None))
        print(f"[ChangeVQASpecialist] Base model: {self.base_model_path}")
        print(f"[ChangeVQASpecialist] LoRA adapter applied: {'yes' if lora_applied else 'no'} (path: {self.adapter_path})")

    def run(self, image_before: Image.Image, image_after: Image.Image, question: str):
        """
        image_before / image_after: PIL Images (bi-temporal pair, im1/im2 style)
        question: natural language change-detection question
        Returns dict matching the same output shape as FusionSpecialist.run()
        """
        # Ensure images match training resolution (128x128) for optimal performance & low VRAM
        if hasattr(image_before, "resize"):
            image_before = image_before.resize((128, 128), Image.BILINEAR)
        if hasattr(image_after, "resize"):
            image_after = image_after.resize((128, 128), Image.BILINEAR)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_before},
                {"type": "image", "image": image_after},
                {"type": "text", "text": question},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=30,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = output.sequences[0][inputs["input_ids"].shape[1]:]
        answer = self.processor.decode(generated_ids, skip_special_tokens=True).strip()

        # NOTE on Confidence Limitation:
        # Fine-tuned LoRA logits are saturated (>0.9999), so confidence values near 1.0
        # should not be interpreted as calibrated certainty -- accuracy (70.5% on the
        # 200-sample CDVQA eval) is the trustworthy metric for this specialist, not
        # per-answer confidence.
        # Confidence: average token probability across generated tokens
        if output.scores:
            probs = []
            for i, score in enumerate(output.scores):
                token_id = generated_ids[i] if i < len(generated_ids) else None
                if token_id is not None:
                    prob = torch.softmax(score[0], dim=-1)[token_id].item()
                    probs.append(prob)
            confidence = sum(probs) / len(probs) if probs else 0.0
        else:
            confidence = 0.0

        return {
            "answer": answer,
            "confidence": confidence,
            "task_type": "change_vqa",
            "evidence_maps": {},  # no spatial evidence for this specialist; text-only output
        }

    def unload(self, force_release: bool = False):
        self.model = None
        self.processor = None
        if force_release:
            release_shared_qwen_vl()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    def memory_footprint_gb(self):
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            if allocated > 0.5:
                return round(allocated, 2)
        return 3.5  # base model (4-bit) + LoRA adapter fallback estimate


# ---------- Smoke test with a real pair from your Val set ----------

if __name__ == "__main__":
    import json

    DATA_DIR = "/home/pavitra/satquery/data/CDVQA"
    IMAGE_DIR = "/home/pavitra/satquery/data/SECOND"

    with open(f"{DATA_DIR}/Val_questions.json") as f:
        questions = json.load(f)["questions"]
    with open(f"{DATA_DIR}/Val_images.json") as f:
        images = json.load(f)["images"]

    img_by_id = {i["id"]: i for i in images}
    sample_q = questions[0]
    sample_img = img_by_id[sample_q["img_id"]]
    file_name = sample_img["file_name"]

    img1 = Image.open(f"{IMAGE_DIR}/im1/{file_name}").convert("RGB")
    img2 = Image.open(f"{IMAGE_DIR}/im2/{file_name}").convert("RGB")

    specialist = ChangeVQASpecialist()
    print("Loading model...")
    specialist.load()
    print(f"Measured memory footprint: {specialist.memory_footprint_gb()} GB")

    print(f"\nQuestion: {sample_q['question']}")
    result = specialist.run(img1, img2, sample_q["question"])
    print("Answer:", result["answer"])
    print("Confidence:", f"{result['confidence']:.4f}")

    specialist.unload()
    print("\nSmoke test complete")
