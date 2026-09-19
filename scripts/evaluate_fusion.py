import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

sys.path.insert(0, "/home/pavitra/satquery/scripts")
from fusion_model import FusionModel, CLASS_NAMES
from fusion_dataset import BenGeDataset, fusion_collate_fn


def evaluate_fusion(
    checkpoint_path="/home/pavitra/satquery/checkpoints/fusion_model.pt",
    threshold=0.5,
    max_samples=None,
    output_json="/home/pavitra/satquery/scripts/results/fusion_eval.json",
):
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("OPTICAL-SAR FUSION MODEL EVALUATION")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Threshold:  {threshold}")
    print("=" * 75)

    # 1. Dataset & Validation Split
    full_dataset = BenGeDataset(max_samples=max_samples)
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len

    _, val_ds = random_split(
        full_dataset, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, collate_fn=fusion_collate_fn)
    print(f"Evaluating on held-out validation set: {len(val_ds)} samples")

    # 2. Load Model
    num_classes = len(CLASS_NAMES)
    model = FusionModel(num_classes=num_classes)
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint {checkpoint_path} not found. Running with initialized weights.")
    model.to(device)
    model.eval()

    # 3. Collect Predictions & Ground Truths
    all_preds = []
    all_targets = []
    all_patch_ids = []

    with torch.no_grad():
        for batch in val_loader:
            sar = batch["sar"].to(device)
            optical = batch["optical"].to(device)
            target = batch["target"].to(device)
            all_patch_ids.extend(batch["patch_id"])

            logits = model(sar, optical)
            probs = torch.sigmoid(logits)

            preds = (probs >= threshold).float()
            all_preds.append(preds.cpu().numpy())
            all_targets.append(target.cpu().numpy())

    all_preds = np.vstack(all_preds)      # (N, num_classes)
    all_targets = np.vstack(all_targets)  # (N, num_classes)
    N = all_targets.shape[0]

    # 4. Per-Class Metrics & Majority Baseline
    print("\n" + "-" * 95)
    print(f"{'Class Name':35s} | {'Support':7s} | {'Prec':6s} | {'Rec':6s} | {'F1':6s} | {'Acc':6s} | {'MajBase':7s} | {'Status'}")
    print("-" * 95)

    class_metrics = {}
    f1_list = []
    prec_list = []
    rec_list = []
    acc_list = []

    for i, cls_name in enumerate(CLASS_NAMES):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        support = int(np.sum(y_true == 1))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / N

        # Majority baseline: always predict the most frequent binary class (0 or 1)
        neg_count = N - support
        majority_baseline = max(support, neg_count) / N

        f1_list.append(f1)
        prec_list.append(precision)
        rec_list.append(recall)
        acc_list.append(accuracy)

        # Flag rare-class instability or near-baseline performance
        status = "OK"
        if support < 15:
            status = "Rare Class (<15)"
        elif accuracy <= majority_baseline:
            status = "Near/At Baseline"

        print(
            f"{cls_name[:35]:35s} | {support:7d} | {precision:5.1%} | {recall:5.1%} | {f1:5.1%} | "
            f"{accuracy:5.1%} | {majority_baseline:6.1%} | {status}"
        )

        class_metrics[cls_name] = {
            "support": support,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "accuracy": round(accuracy, 4),
            "majority_baseline": round(majority_baseline, 4),
            "status": status,
        }

    macro_f1 = float(np.mean(f1_list))
    macro_prec = float(np.mean(prec_list))
    macro_rec = float(np.mean(rec_list))
    mean_acc = float(np.mean(acc_list))

    print("-" * 95)
    print(f"{'OVERALL MACRO AVERAGE':35s} | {N:7d} | {macro_prec:5.1%} | {macro_rec:5.1%} | {macro_f1:5.1%} | {mean_acc:5.1%} |")
    print("=" * 95)

    results_payload = {
        "summary": {
            "samples_evaluated": N,
            "threshold": threshold,
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "mean_accuracy": round(mean_acc, 4),
        },
        "per_class": class_metrics,
    }

    with open(output_json, "w") as f:
        json.dump(results_payload, f, indent=2)

    print(f"\nEvaluation results saved to: {output_json}")

    # Standardize predictions to match eval/schemas/eval_io.py
    try:
        from eval.schemas.eval_io import EvalResults, ModelPrediction
        model_predictions = []
        for i, pid in enumerate(all_patch_ids):
            pred_labels = [CLASS_NAMES[c] for c in range(num_classes) if all_preds[i, c] == 1]
            ans = ", ".join(sorted(pred_labels)) if pred_labels else "no significant land-cover features detected"
            model_predictions.append(
                ModelPrediction(
                    sample_id=str(pid),
                    task="vqa",
                    model_name="fusion_dual_cnn",
                    answer=ans,
                )
            )

        eval_results = EvalResults(
            model_name="fusion_dual_cnn",
            benchmark="ben_ge_8k",
            predictions=model_predictions,
        )
        eval_preds_path = "/home/pavitra/satquery/eval/predictions/fusion_dual_cnn_ben_ge_8k_vqa.json"
        os.makedirs(os.path.dirname(eval_preds_path), exist_ok=True)
        with open(eval_preds_path, "w") as f:
            json.dump(eval_results.model_dump(), f, indent=2)
        print(f"Saved standardized EvalResults predictions to: {eval_preds_path}\n")
    except Exception as e:
        print(f"Warning: Could not save standardized EvalResults: {e}")

    return results_payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="/home/pavitra/satquery/checkpoints/fusion_model.pt")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="/home/pavitra/satquery/scripts/results/fusion_eval.json")
    args = parser.parse_args()

    evaluate_fusion(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        output_json=args.output,
    )
