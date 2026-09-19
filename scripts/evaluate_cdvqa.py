import os
import sys
import json
import argparse
from collections import Counter, defaultdict
from PIL import Image
import torch
from transformers import (
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
)
from peft import PeftModel

sys.path.insert(0, "/home/pavitra/satquery/scripts")
sys.path.insert(0, "/home/pavitra/satquery")
from cdvqa_dataset import load_cdvqa_split


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL LoRA on CDVQA")
    parser.add_argument(
        "--base_model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="Base model ID or path",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default="/home/pavitra/satquery/checkpoints/qwen2.5-vl-cdvqa-lora",
        help="Path to trained LoRA adapter",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="Val",
        help="Dataset split to evaluate (Val, Test, Test2)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of samples to evaluate (e.g. 200 or -1 for full)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/home/pavitra/satquery/scripts/results/val_eval_200.json",
        help="Path to save evaluation JSON results",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=128,
        help="Image resize dimension (must match training resolution 128)",
    )
    return parser.parse_args()


def normalize_text(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = text.rstrip(".").rstrip(",")
    return text


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    print("=" * 60)
    print("CDVQA EVALUATION")
    print(f"Base Model:    {args.base_model}")
    print(f"Adapter Path:  {args.adapter_path}")
    print(f"Split:         {args.split}")
    print(f"Limit:         {args.limit if args.limit > 0 else 'FULL'}")
    print(f"Image Size:    {args.image_size}x{args.image_size}")
    print("=" * 60)

    # 1. Load Dataset
    print(f"\n[1/4] Loading {args.split} split annotations...")
    all_samples = load_cdvqa_split(args.split)
    total_samples = len(all_samples)
    print(f"Total valid samples in {args.split}: {total_samples}")

    # Compute Majority Baseline across entire split
    type_answers_full = defaultdict(Counter)
    for s in all_samples:
        type_answers_full[s["question_type"]][normalize_text(s["answer"])] += 1

    majority_per_type_full = {
        qtype: counts.most_common(1)[0][0]
        for qtype, counts in type_answers_full.items()
    }
    majority_correct_full = sum(
        1 for s in all_samples
        if normalize_text(s["answer"]) == majority_per_type_full[s["question_type"]]
    )
    full_majority_baseline = (majority_correct_full / total_samples) * 100.0

    print("\nFull Split Majority Class per Question Type:")
    for qtype, maj_ans in majority_per_type_full.items():
        type_total = len([s for s in all_samples if s["question_type"] == qtype])
        top_count = type_answers_full[qtype][maj_ans]
        print(f"  - {qtype:20s}: '{maj_ans}' ({top_count}/{type_total} = {100.0 * top_count / type_total:.2f}%)")
    print(f"Full {args.split} Split Majority Baseline Accuracy: {full_majority_baseline:.2f}%\n")

    # Select subset if limited
    if args.limit > 0 and args.limit < total_samples:
        eval_samples = all_samples[:args.limit]
    else:
        eval_samples = all_samples

    # Compute Majority Baseline for evaluated subset
    type_answers_eval = defaultdict(Counter)
    for s in eval_samples:
        type_answers_eval[s["question_type"]][normalize_text(s["answer"])] += 1

    majority_per_type_eval = {
        qtype: counts.most_common(1)[0][0]
        for qtype, counts in type_answers_eval.items()
    }
    subset_majority_correct = sum(
        1 for s in eval_samples
        if normalize_text(s["answer"]) == majority_per_type_eval[s["question_type"]]
    )
    subset_majority_baseline = (subset_majority_correct / len(eval_samples)) * 100.0
    print(f"Evaluated Subset ({len(eval_samples)} samples) Majority Baseline: {subset_majority_baseline:.2f}%\n")

    # 2. Load Model & Processor
    print("[2/4] Loading model and LoRA adapter with 4-bit quantization...")
    processor = AutoProcessor.from_pretrained(args.adapter_path)
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=quantization_config,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    print("Model + LoRA adapter loaded successfully into VRAM.\n")

    # 3. Inference Loop
    print(f"[3/4] Running inference on {len(eval_samples)} samples...")
    results = []
    skipped_images = 0
    type_metrics = defaultdict(lambda: {"total": 0, "correct_substr": 0, "correct_exact": 0})
    overall_correct_substr = 0
    overall_correct_exact = 0

    for idx, s in enumerate(eval_samples):
        q = s["question"]
        gt = s["answer"]
        qtype = s["question_type"]
        im1_path = s["im1_path"]
        im2_path = s["im2_path"]

        try:
            im1 = Image.open(im1_path).convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR)
            im2 = Image.open(im2_path).convert("RGB").resize((args.image_size, args.image_size), Image.BILINEAR)
        except Exception as e:
            skipped_images += 1
            print(f"Skipping sample {idx} due to image load error: {e}")
            continue

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "image"},
                    {"type": "text", "text": q},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[prompt],
            images=[[im1, im2]],
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=False,
            )

        generated_ids = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        pred = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0].strip()

        norm_gt = normalize_text(gt)
        norm_pred = normalize_text(pred)

        # Case-insensitive substring match
        is_match_substr = norm_gt in norm_pred
        # Exact match
        is_match_exact = (norm_gt == norm_pred)

        if is_match_substr:
            overall_correct_substr += 1
            type_metrics[qtype]["correct_substr"] += 1

        if is_match_exact:
            overall_correct_exact += 1
            type_metrics[qtype]["correct_exact"] += 1

        type_metrics[qtype]["total"] += 1

        results.append({
            "idx": idx,
            "question": q,
            "question_type": qtype,
            "ground_truth": gt,
            "prediction": pred,
            "match_substr": is_match_substr,
            "match_exact": is_match_exact,
            "file_name": s.get("file_name", ""),
        })

        if (idx + 1) % 25 == 0 or (idx + 1) == len(eval_samples):
            current_eval_count = len(results)
            cur_acc = (overall_correct_substr / current_eval_count) * 100.0
            print(f"  [{idx + 1}/{len(eval_samples)}] Current Substring Acc: {cur_acc:.2f}% ({overall_correct_substr}/{current_eval_count})")

    # 4. Reporting & Saving
    eval_count = len(results)
    final_acc_substr = (overall_correct_substr / eval_count) * 100.0 if eval_count > 0 else 0.0
    final_acc_exact = (overall_correct_exact / eval_count) * 100.0 if eval_count > 0 else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Samples Evaluated:   {eval_count}")
    print(f"Skipped Images:            {skipped_images}")
    print(f"Overall Accuracy (Substr): {final_acc_substr:.2f}% ({overall_correct_substr}/{eval_count})")
    print(f"Overall Accuracy (Exact):  {final_acc_exact:.2f}% ({overall_correct_exact}/{eval_count})")
    print(f"Evaluated Majority Base:   {subset_majority_baseline:.2f}%")
    print(f"Full Split Majority Base:  {full_majority_baseline:.2f}%")
    print(f"Delta vs Eval Majority:    {final_acc_substr - subset_majority_baseline:+.2f} percentage points")
    print(f"Delta vs Full Majority:    {final_acc_substr - full_majority_baseline:+.2f} percentage points")
    print("-" * 60)
    print(f"{'Question Type':22s} | {'Count':5s} | {'Substr Acc':10s} | {'Exact Acc':10s} | {'Top Answer':12s} | {'Maj Base'}")
    print("-" * 75)

    type_summary = {}
    for qtype in sorted(type_metrics.keys()):
        m = type_metrics[qtype]
        cnt = m["total"]
        sub_acc = (m["correct_substr"] / cnt) * 100.0 if cnt > 0 else 0.0
        ex_acc = (m["correct_exact"] / cnt) * 100.0 if cnt > 0 else 0.0
        top_ans = majority_per_type_eval.get(qtype, "N/A")
        top_cnt = type_answers_eval[qtype][top_ans]
        maj_acc = (top_cnt / cnt) * 100.0 if cnt > 0 else 0.0
        print(f"{qtype:22s} | {cnt:5d} | {sub_acc:9.2f}% | {ex_acc:9.2f}% | {top_ans:12s} | {maj_acc:6.2f}%")
        type_summary[qtype] = {
            "count": cnt,
            "substring_accuracy": sub_acc,
            "exact_accuracy": ex_acc,
            "majority_answer": top_ans,
            "majority_baseline": maj_acc,
        }

    output_payload = {
        "metadata": {
            "split": args.split,
            "samples_evaluated": eval_count,
            "skipped_images": skipped_images,
            "image_size": [args.image_size, args.image_size],
            "overall_accuracy_substring": final_acc_substr,
            "overall_accuracy_exact": final_acc_exact,
            "subset_majority_baseline": subset_majority_baseline,
            "full_split_majority_baseline": full_majority_baseline,
            "delta_vs_subset_majority": final_acc_substr - subset_majority_baseline,
            "delta_vs_full_majority": final_acc_substr - full_majority_baseline,
        },
        "per_question_type": type_summary,
        "predictions": results,
    }

    with open(args.output_file, "w") as f:
        json.dump(output_payload, f, indent=2)

    print(f"\nDetailed per-sample predictions saved to: {args.output_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
