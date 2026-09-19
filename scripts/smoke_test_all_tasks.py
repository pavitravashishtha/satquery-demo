"""
smoke_test_all_tasks.py — Comprehensive Smoke Test for SatQuery AI.

Runs one real query through the full pipeline (orchestrator.run_query)
for each of the 5 supported task types:
  1. vqa
  2. captioning
  3. grounding
  4. change_vqa
  5. fusion_analysis

Using real images from data/SECOND and data/ben-ge-8k.
Outputs full JSON-formatted dictionaries for each query.
"""

import os
import sys
import json
import time
import pprint

# Ensure scripts directory is in sys.path
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from orchestrator import run_query
from model_registry import build_registry


def run_smoke_tests():
    print("=" * 70)
    print("SATQUERY AI — 5-TASK PIPELINE SMOKE TEST")
    print("=" * 70)

    # Base image paths
    second_im1 = "/home/pavitra/satquery/data/SECOND/im1/00003.png"
    second_im2 = "/home/pavitra/satquery/data/SECOND/im2/00003.png"
    benge_opt = "/home/pavitra/satquery/data/ben-ge-8k/sentinel-2/S2A_MSIL2A_20171002T094031_64_46/S2A_MSIL2A_20171002T094031_64_46_B04.tif"
    benge_sar = "/home/pavitra/satquery/data/ben-ge-8k/sentinel-1/S1B_IW_GRDH_1SDV_20170929T045327_34TCR_64_46/S1B_IW_GRDH_1SDV_20170929T045327_34TCR_64_46_VV.tif"

    # Verify input images exist
    for p in [second_im1, second_im2, benge_opt, benge_sar]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required smoke test image: {p}")

    test_cases = [
        {
            "task_type": "vqa",
            "query": "What does the land cover look like around Chennai?",
            "images": [second_im2],
            "description": "Optical single-image Visual Question Answering via GeneralVLMSpecialist (Qwen2.5-VL base)",
        },
        {
            "task_type": "captioning",
            "query": "Describe this aerial image in detail.",
            "images": [second_im2],
            "description": "Optical single-image Scene Captioning via GeneralVLMSpecialist (Qwen2.5-VL base)",
        },
        {
            "task_type": "grounding",
            "query": "Locate and highlight all the buildings in this scene.",
            "images": [second_im2],
            "description": "Optical single-image Visual Grounding & Bounding Boxes via GeneralVLMSpecialist (Qwen2.5-VL base)",
        },
        {
            "task_type": "change_vqa",
            "query": "Did the buildings change between the before and after images?",
            "images": {
                "before": second_im1,
                "after": second_im2,
            },
            "description": "Bi-temporal Change Detection VQA via fine-tuned Qwen-CDVQA LoRA",
        },
        {
            "task_type": "fusion_analysis",
            "query": "Combine optical and SAR imagery to identify flooded areas in Assam during cloud cover.",
            "images": {
                "optical": benge_opt,
                "sar": benge_sar,
            },
            "description": "Multi-modal Optical-SAR Fusion Analysis via 2-Branch CNN + Grad-CAM",
        },
    ]

    all_results = {}

    for idx, tc in enumerate(test_cases, 1):
        task = tc["task_type"]
        query = tc["query"]
        images = tc["images"]
        desc = tc["description"]

        print(f"\n[{idx}/5] SMOKE TEST: {task.upper()}")
        print(f"  Description: {desc}")
        print(f"  Query: \"{query}\"")
        if isinstance(images, dict):
            print(f"  Images (dict):")
            for k, v in images.items():
                print(f"    - {k}: {v}")
        else:
            print(f"  Images (list): {images}")

        start_time = time.time()
        try:
            result = run_query(raw_query=query, images=images)
            duration = time.time() - start_time
            print(f"  Status: SUCCESS ({duration:.2f}s)")
            print(f"  Interpreted: {result.get('interpreted')}")
            print(f"  Needs Clarification: {result.get('needs_clarification')}")
            print(f"  Specialist Results Count: {len(result.get('results', []))}")
            for res_item in result.get("results", []):
                print(f"    - Task: {res_item.get('task_type')}, Specialist: {res_item.get('specialist')}")
                out = res_item.get("output", {})
                print(f"      Answer: {out.get('answer')}")
                print(f"      Confidence: {out.get('confidence')}")
                if "evidence_maps" in out and out["evidence_maps"]:
                    if isinstance(out["evidence_maps"], dict) and "boxes" in out["evidence_maps"]:
                        print(f"      Grounding Boxes: {out['evidence_maps']['boxes']}")
                    else:
                        print(f"      Evidence Map Keys: {list(out['evidence_maps'].keys())}")

            all_results[task] = {
                "success": True,
                "query": query,
                "images": images,
                "duration_seconds": duration,
                "result": result,
                "error": None,
            }
        except Exception as e:
            import traceback
            duration = time.time() - start_time
            err_tb = traceback.format_exc()
            print(f"  Status: FAILED ({duration:.2f}s)")
            print(f"  Error: {e}")
            print(f"  Traceback:\n{err_tb}")
            all_results[task] = {
                "success": False,
                "query": query,
                "images": images,
                "duration_seconds": duration,
                "result": None,
                "error": str(e),
                "traceback": err_tb,
            }

    # Save summary report to JSON
    output_file = "/home/pavitra/satquery/scripts/smoke_test_results.json"
    with open(output_file, "w") as f:
        # Convert non-serializable objects (like NumPy arrays or enums) if any
        def default_serializer(obj):
            if hasattr(obj, "tolist"):
                return obj.tolist()
            if hasattr(obj, "value"):
                return obj.value
            return str(obj)
        json.dump(all_results, f, indent=2, default=default_serializer)

    print("\n" + "=" * 70)
    print(f"SMOKE TEST COMPLETE. Raw results saved to: {output_file}")
    all_passed = all(r["success"] for r in all_results.values())
    print(f"OVERALL STATUS: {'ALL 5 PASSED' if all_passed else 'SOME FAILED'}")
    print("=" * 70)
    return all_results


if __name__ == "__main__":
    run_smoke_tests()
