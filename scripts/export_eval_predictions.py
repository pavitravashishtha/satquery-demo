import os
import sys
import time
import json
import argparse
from PIL import Image

# Ensure project root and scripts are in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
SCRIPTS_DIR = os.path.join(PROJECT_ROOT, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from eval.schemas.eval_io import ModelPrediction, EvalResults
from cdvqa_dataset import load_cdvqa_split


# Task name mapping translating internal specialist task_type to eval_io.py ModelPrediction.task Literal
TASK_TYPE_MAPPING = {
    "captioning": "caption",
    "optical_sar_fusion": "fusion_vqa",
    "fusion_analysis": "fusion_vqa",
    "change_vqa": "change_vqa",
    "vqa": "vqa",
    "grounding": "grounding",
    "caption": "caption",
    "fusion_vqa": "fusion_vqa",
}


def map_task_type(task_type: str) -> str:
    """Translates specialist task_type to eval_io.py schema task literal."""
    mapped = TASK_TYPE_MAPPING.get(task_type)
    if not mapped:
        raise ValueError(f"Unrecognized task_type '{task_type}'. Valid mappings: {list(TASK_TYPE_MAPPING.keys())}")
    return mapped


def run_export(
    specialist_name: str = "geochat",
    benchmark: str = "cdvqa",
    split: str = "Val",
    max_samples: int = 20,
    output_path: str = None,
):
    print("=" * 70)
    print("SATQUERY AI EVALUATION EXPORTER")
    print(f"Specialist:  {specialist_name}")
    print(f"Benchmark:   {benchmark}")
    print(f"Split:       {split}")
    print(f"Max Samples: {max_samples}")
    print("=" * 70)

    # 1. Load dataset split
    if benchmark.lower() == "cdvqa":
        samples = load_cdvqa_split(split)
        if max_samples and max_samples > 0:
            samples = samples[:max_samples]
        print(f"Loaded {len(samples)} samples from CDVQA {split} split.")
    else:
        raise NotImplementedError(f"Benchmark '{benchmark}' data loading not yet implemented.")

    # 2. Instantiate and load specialist
    if specialist_name.lower() in ["geochat", "geochat-7b"]:
        from geochat_specialist import GeoChatSpecialist
        specialist = GeoChatSpecialist()
        model_name = "geochat-7b"
        default_task = "vqa"
    elif specialist_name.lower() in ["qwen", "qwen_change_vqa"]:
        from qwen_specialist import ChangeVQASpecialist
        specialist = ChangeVQASpecialist()
        model_name = "qwen2.5-vl-cdvqa"
        default_task = "change_vqa"
    elif specialist_name.lower() in ["fusion", "fusion_specialist"]:
        from fusion_model import FusionSpecialist
        specialist = FusionSpecialist()
        model_name = "optical-sar-fusion"
        default_task = "fusion_vqa"
    else:
        raise ValueError(f"Unknown specialist '{specialist_name}'. Choose 'geochat', 'qwen', or 'fusion'.")

    specialist.load()

    # 3. Run inference across samples
    predictions = []
    print(f"\nRunning inference on {len(samples)} samples...")
    t_start = time.time()

    for idx, sample in enumerate(samples):
        t0 = time.time()
        
        if specialist_name.lower() in ["geochat", "geochat-7b"]:
            # GeoChat: single-image VQA on post-event image (im2)
            res = specialist.run(
                image=sample["im2_path"],
                query=sample["question"],
                task_type="vqa",
            )
            raw_task = res.get("task_type", "vqa")
        elif specialist_name.lower() in ["qwen", "qwen_change_vqa"]:
            # Qwen: bi-temporal change VQA
            im1 = Image.open(sample["im1_path"]).convert("RGB")
            im2 = Image.open(sample["im2_path"]).convert("RGB")
            res = specialist.run(
                image_before=im1,
                image_after=im2,
                question=sample["question"],
            )
            raw_task = res.get("task_type", "change_vqa")
        else:
            raise NotImplementedError("Fusion export requires paired SAR/Optical benchmark.")

        elapsed = time.time() - t0
        mapped_task = map_task_type(raw_task)

        pred = ModelPrediction(
            sample_id=str(sample["question_id"]),
            task=mapped_task,
            answer=res.get("answer", "").strip(),
            model_name=model_name,
            runtime_sec=round(elapsed, 4),
        )
        predictions.append(pred)

        print(f"[{idx + 1:2d}/{len(samples)}] sample_id={pred.sample_id} | "
              f"Pred: '{pred.answer[:30]}' | "
              f"GT: '{sample['answer']}' | "
              f"time: {elapsed:.2f}s")

    total_time = time.time() - t_start
    print(f"\nInference completed in {total_time:.2f}s ({total_time / max(len(samples), 1):.2f}s/sample)")

    # 4. Unload specialist cleanly
    specialist.unload()

    # 5. Build and validate EvalResults
    eval_results = EvalResults(
        model_name=model_name,
        benchmark=benchmark.lower(),
        predictions=predictions,
    )

    # 6. Save output JSON
    if output_path is None:
        os.makedirs(os.path.join(PROJECT_ROOT, "eval", "predictions"), exist_ok=True)
        output_path = os.path.join(
            PROJECT_ROOT, "eval", "predictions", f"{model_name}_{benchmark.lower()}_{mapped_task}.json"
        )
    else:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(eval_results.model_dump_json(indent=2))

    print(f"\n✓ Exported {len(predictions)} predictions to: {output_path}")
    print("✓ Output verified and validated cleanly against EvalResults schema!")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export specialist predictions to SatQuery AI EvalResults format.")
    parser.add_argument("--specialist", type=str, default="geochat", help="Specialist to run ('geochat', 'qwen', 'fusion')")
    parser.add_argument("--benchmark", type=str, default="cdvqa", help="Benchmark name (e.g. 'cdvqa')")
    parser.add_argument("--split", type=str, default="Val", help="Dataset split ('Val', 'Test')")
    parser.add_argument("--max-samples", type=int, default=20, help="Number of samples to export (0 for all)")
    parser.add_argument("--output", type=str, default=None, help="Custom output JSON path")
    args = parser.parse_args()

    run_export(
        specialist_name=args.specialist,
        benchmark=args.benchmark,
        split=args.split,
        max_samples=args.max_samples,
        output_path=args.output,
    )
