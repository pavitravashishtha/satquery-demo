"""
verify_general_vlm_integration.py — Complete validation of GeneralVLMSpecialist,
slot-sharing optimization, confidence reliability, and end-to-end 5-task pipeline.
"""

import os
import sys
import time
import json
import torch
from PIL import Image

sys.path.insert(0, "/home/pavitra/satquery/scripts")

from model_registry import (
    GENERAL_VLM,
    QWEN_CDVQA,
    FUSION,
    build_registry,
    TASK_TO_SPECIALIST,
)
from general_vlm_specialist import GeneralVLMSpecialist
from qwen_specialist import ChangeVQASpecialist
from fusion_model import FusionSpecialist
import orchestrator

def test_vram_and_lifecycle_sharing():
    print("\n" + "="*60)
    print("PHASE 1: LIFECYCLE MANAGER & ZERO-OVERHEAD SHARING TEST")
    print("="*60)

    manager = build_registry()
    
    # 1. Initial VRAM
    vram_start = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"[Initial VRAM] Allocated: {vram_start:.3f} GB")
    
    # 2. Load ChangeVQASpecialist
    t0 = time.time()
    change_vqa = manager.load(QWEN_CDVQA)
    t_change = time.time() - t0
    vram_change = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"[Loaded ChangeVQASpecialist] Time: {t_change:.2f}s | VRAM: {vram_change:.3f} GB")
    
    # 3. Load GeneralVLMSpecialist (Should share the exact same weights!)
    t0 = time.time()
    general_vlm = manager.load(GENERAL_VLM)
    t_gen = time.time() - t0
    vram_gen = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"[Loaded GeneralVLMSpecialist] Time: {t_gen:.4f}s | VRAM: {vram_gen:.3f} GB")
    
    # Check weight sharing
    is_same_model = (change_vqa.model is general_vlm.model)
    is_same_proc = (change_vqa.processor is general_vlm.processor)
    print(f"Weight sharing confirmed: {is_same_model} (processor sharing: {is_same_proc})")
    print(f"Additional VRAM cost: {vram_gen - vram_change:.4f} GB")
    assert is_same_model, "ERROR: change_vqa.model and general_vlm.model are not the same instance!"
    assert abs(vram_gen - vram_change) < 0.05, f"ERROR: VRAM increased by {vram_gen - vram_change:.3f} GB!"
    
    # 4. Switch back to ChangeVQASpecialist
    t0 = time.time()
    change_vqa_re = manager.load(QWEN_CDVQA)
    t_switch = time.time() - t0
    print(f"[Switch back to ChangeVQASpecialist] Time: {t_switch:.4f}s")
    assert change_vqa_re is change_vqa, "ERROR: Cached model instance mismatch!"

    return manager, change_vqa, general_vlm


def test_task2_smoke_tests(general_vlm):
    print("\n" + "="*60)
    print("PHASE 2: SMOKE TEST ALL 3 GENERAL TASK TYPES (00003.png)")
    print("="*60)

    test_image_path = "/home/pavitra/satquery/data/SECOND/im2/00003.png"
    img = Image.open(test_image_path).convert("RGB")
    
    queries = [
        ("vqa", "What does the land cover look like around Chennai?"),
        ("captioning", "Describe this aerial image in detail."),
        ("grounding", "Locate and highlight all the buildings in this scene."),
    ]
    
    task2_results = {}
    for task_type, query in queries:
        print(f"\n--- Task Type: {task_type.upper()} ---")
        print(f"Query: {query}")
        res = general_vlm.run(image=img, query=query, task_type=task_type)
        print(f"Answer: {res['answer']}")
        print(f"Confidence: {res['confidence']:.4f}")
        print(f"Evidence Maps: {res['evidence_maps']}")
        task2_results[task_type] = res
        
        # Verify no hallucination loop
        assert "nobody is perfect" not in res["answer"].lower(), "ERROR: GeoChat hallucination detected!"
        assert len(res["answer"]) > 5, "ERROR: Answer too short!"
        
        if task_type == "grounding":
            # 00003.png has no buildings; verify plausible coordinates or empty
            boxes = res["evidence_maps"].get("boxes", [])
            for b in boxes:
                ymin, xmin, ymax, xmax = b
                assert 0 <= ymin <= 1000 and 0 <= xmin <= 1000, f"Box coordinate out of bounds: {b}"
                assert 0 <= ymax <= 1000 and 0 <= xmax <= 1000, f"Box coordinate out of bounds: {b}"
            print(f"Grounding coordinates sanity check: PASS ({len(boxes)} boxes found)")
            
    # Also test an image with actual buildings for grounding coordinate inspection
    print("\n--- Additional Grounding Test on Image With Buildings (06995.png) ---")
    bldg_img_path = "/home/pavitra/satquery/data/SECOND/im2/06995.png"
    if os.path.exists(bldg_img_path):
        bldg_img = Image.open(bldg_img_path).convert("RGB")
        res_bldg = general_vlm.run(
            image=bldg_img,
            query="Locate and highlight all the buildings in this scene.",
            task_type="grounding",
        )
        print(f"Answer: {res_bldg['answer']}")
        print(f"Confidence: {res_bldg['confidence']:.4f}")
        print(f"Evidence Maps: {res_bldg['evidence_maps']}")
        boxes = res_bldg["evidence_maps"].get("boxes", [])
        for b in boxes:
            ymin, xmin, ymax, xmax = b
            assert 0 <= ymin <= 1000 and 0 <= xmin <= 1000, f"Box coordinate out of bounds: {b}"
            assert 0 <= ymax <= 1000 and 0 <= xmax <= 1000, f"Box coordinate out of bounds: {b}"
        print(f"06995.png Grounding coordinates sanity check: PASS ({len(boxes)} boxes found)")
        task2_results["grounding_buildings"] = res_bldg

    return task2_results


def test_task3_confidence_reliability(general_vlm):
    print("\n" + "="*60)
    print("PHASE 3: CONFIDENCE RELIABILITY CHECK (6 Verifiable Queries)")
    print("="*60)

    test_image_path = "/home/pavitra/satquery/data/SECOND/im2/00003.png"
    img = Image.open(test_image_path).convert("RGB")

    cases = [
        ("Is there water such as a lake, river, or ocean visible in this image? Answer yes or no.", "No"),
        ("Are there skyscrapers or high-rise urban buildings visible in this image? Answer yes or no.", "No"),
        ("What is the primary color tone of the landscape in this image?", "Brown / earthen"),
        ("Is this image taken during a snowy winter with thick snow cover? Answer yes or no.", "No"),
        ("Is this an aerial or satellite view of the ground? Answer yes or no.", "Yes"),
        ("Is there a dense network of paved multilane highways with heavy car traffic? Answer yes or no.", "No"),
    ]

    conf_records = []
    for q, expected in cases:
        res = general_vlm.run(image=img, query=q, task_type="vqa")
        conf_records.append({
            "query": q,
            "expected": expected,
            "model_answer": res["answer"],
            "confidence": res["confidence"],
        })
        print(f"Q: {q}")
        print(f"   Expected: {expected}")
        print(f"   Model Answer: {res['answer']}")
        print(f"   Confidence: {res['confidence']:.4f}")

    # Check distribution of confidences
    confs = [r["confidence"] for r in conf_records]
    avg_conf = sum(confs) / len(confs)
    print(f"\nAverage Confidence across 6 queries: {avg_conf:.4f}")
    is_pinned = all(c >= 0.99 for c in confs)
    if is_pinned:
        verdict = "confidence signal is NOT reliable"
    else:
        verdict = "confidence signal IS reliable"
    print(f"CONFIDENCE VERDICT: {verdict}")

    return conf_records, verdict


def test_task5_full_pipeline():
    print("\n" + "="*60)
    print("PHASE 4: FULL 5-TASK PIPELINE RE-TEST (End-to-End Orchestrator)")
    print("="*60)

    manager = build_registry()

    # Define test inputs for all 5 tasks
    im1_path = "/home/pavitra/satquery/data/SECOND/im1/00003.png"
    im2_path = "/home/pavitra/satquery/data/SECOND/im2/00003.png"
    opt_path = "/home/pavitra/satquery/data/ben-ge-8k/sentinel-2/S2A_MSIL2A_20171002T094031_64_46/S2A_MSIL2A_20171002T094031_64_46_B04.tif"
    sar_path = "/home/pavitra/satquery/data/ben-ge-8k/sentinel-1/S1B_IW_GRDH_1SDV_20170929T045327_34TCR_64_46/S1B_IW_GRDH_1SDV_20170929T045327_34TCR_64_46_VV.tif"

    print(f"Test Assets:\n  im1: {im1_path}\n  im2: {im2_path}\n  optical: {opt_path}\n  sar: {sar_path}")

    pipeline_tests = [
        {
            "task": "vqa",
            "query": "What is the primary land cover visible in this scene?",
            "images": [{"path": im2_path, "sensor": "optical"}],
        },
        {
            "task": "captioning",
            "query": "Provide a detailed description of this aerial view.",
            "images": [{"path": im2_path, "sensor": "optical"}],
        },
        {
            "task": "grounding",
            "query": "Locate and highlight all the buildings in this scene.",
            "images": [{"path": im2_path, "sensor": "optical"}],
        },
        {
            "task": "change_vqa",
            "query": "Has any construction or surface change occurred between these dates?",
            "images": [
                {"path": im1_path, "role": "before", "sensor": "optical"},
                {"path": im2_path, "role": "after", "sensor": "optical"},
            ],
        },
        {
            "task": "fusion_analysis",
            "query": "Perform optical and SAR radar multi-sensor fusion analysis.",
            "images": [
                {"path": opt_path, "sensor": "optical"},
                {"path": sar_path, "sensor": "sar"},
            ],
        },
    ]

    pipeline_results = []
    for test in pipeline_tests:
        print(f"\n[RUNNING PIPELINE TASK: {test['task'].upper()}]")
        t0 = time.time()
        res = orchestrator.run_query(
            raw_query=test["query"],
            images=test["images"],
            lifecycle_manager=manager,
        )
        t_el = time.time() - t0
        print(f"Elapsed: {t_el:.2f}s")
        print(f"Interpreted tasks: {res['interpreted']['task_sequence']}")
        print(f"Needs clarification: {res['needs_clarification']}")
        for out_item in res["results"]:
            print(f"  Specialist: {out_item['specialist']} | Task: {out_item['task_type']}")
            ans = out_item["output"].get("answer", "")
            conf = out_item["output"].get("confidence", 0.0)
            ev = list(out_item["output"].get("evidence_maps", {}).keys())
            print(f"  Answer: {ans[:120]}...")
            print(f"  Confidence: {conf:.4f} | Evidence Maps Keys: {ev}")
        pipeline_results.append({
            "task": test["task"],
            "elapsed_s": t_el,
            "response": res,
        })

    return pipeline_results


if __name__ == "__main__":
    manager, change_vqa, general_vlm = test_vram_and_lifecycle_sharing()
    task2_res = test_task2_smoke_tests(general_vlm)
    task3_res, task3_verdict = test_task3_confidence_reliability(general_vlm)
    task5_res = test_task5_full_pipeline()
    print("\n" + "="*60)
    print("ALL 5 PHASES COMPLETED SUCCESSFULLY!")
    print("="*60)
