import os
import sys
import torch

sys.path.insert(0, "/home/pavitra/satquery/scripts")
from fusion_model import FusionSpecialist
from fusion_dataset import BenGeDataset


def verify_fusion_specialist():
    print("=" * 70)
    print("BOX 4 VERIFICATION: FUSION SPECIALIST LIFECYCLE & INFERENCE")
    print("=" * 70)

    checkpoint_path = "/home/pavitra/satquery/checkpoints/fusion_model.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint {checkpoint_path} not found yet.")

    # 1. Instantiate
    specialist = FusionSpecialist(checkpoint_path=checkpoint_path)
    mem_before = specialist.memory_footprint_gb()
    print(f"Initial GPU memory: {mem_before:.4f} GB")

    # 2. Load
    specialist.load()
    mem_loaded = specialist.memory_footprint_gb()
    print(f"Post-load GPU memory: {mem_loaded:.4f} GB")

    # 3. Load a real validation sample
    dataset = BenGeDataset(max_samples=10)
    sample = dataset[0]
    sar = sample["sar"]
    optical = sample["optical"]
    gt_labels = sample["labels"]

    print(f"\nTest Sample: {sample['patch_id']}")
    print(f"Ground truth labels: {gt_labels}")
    print(f"Input SAR tensor shape:     {sar.shape}")
    print(f"Input Optical tensor shape: {optical.shape}")

    # 4. Run inference
    result = specialist.run(sar, optical, threshold=0.3)
    print("\nInference Output:")
    print(f"  Task Type:    {result['task_type']}")
    print(f"  Confidence:   {result['confidence']:.4f}")
    print(f"  Answer:       {result['answer']}")
    print(f"  Evidence Maps Detected: {list(result['evidence_maps'].keys())}")

    for cls_name, hmap in result["evidence_maps"].items():
        print(f"    - '{cls_name}' heatmap shape: {hmap.shape}, min: {hmap.min():.3f}, max: {hmap.max():.3f}")
        assert hmap.shape == (128, 128), f"Expected (128, 128), got {hmap.shape}"
        assert 0.0 <= hmap.min() and hmap.max() <= 1.0, "Heatmap values must be normalized to [0, 1]"

    # 5. Unload
    print("\nUnloading model...")
    specialist.unload()
    mem_after = specialist.memory_footprint_gb()
    print(f"Post-unload GPU memory: {mem_after:.4f} GB")
    assert specialist.model is None, "Model reference should be cleared"

    print("\n✓ FusionSpecialist interface and lifecycle verified successfully!")
    print("=" * 70)


if __name__ == "__main__":
    verify_fusion_specialist()
