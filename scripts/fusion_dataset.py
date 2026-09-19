import os
import json
import glob
import numpy as np
import csv
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

# Official BigEarthNet 19-class taxonomy
BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# Official 43-to-19 mapping dictionary developed by TU Berlin
BIGEARTHNET_43_TO_19 = {
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Industrial or commercial units": "Industrial or commercial units",
    "Road and rail networks and associated land": "Industrial or commercial units",
    "Port areas": "Industrial or commercial units",
    "Airports": "Industrial or commercial units",
    "Mineral extraction sites": "Industrial or commercial units",
    "Dump sites": "Industrial or commercial units",
    "Construction sites": "Industrial or commercial units",
    "Green urban areas": "Urban fabric",
    "Sport and leisure facilities": "Urban fabric",
    "Non-irrigated arable land": "Arable land",
    "Permanently irrigated land": "Arable land",
    "Rice fields": "Arable land",
    "Vineyards": "Permanent crops",
    "Fruit trees and berry plantations": "Permanent crops",
    "Olive groves": "Permanent crops",
    "Pastures": "Pastures",
    "Annual crops associated with permanent crops": "Arable land",
    "Complex cultivation patterns": "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation": "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas": "Agro-forestry areas",
    "Broad-leaved forest": "Broad-leaved forest",
    "Coniferous forest": "Coniferous forest",
    "Mixed forest": "Mixed forest",
    "Natural grasslands": "Natural grassland and sparsely vegetated areas",
    "Moors and heathland": "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation": "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland-shrub": "Transitional woodland, shrub",
    "Beaches, dunes, sands": "Beaches, dunes, sands",
    "Bare rocks": "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas": "Natural grassland and sparsely vegetated areas",
    "Burnt areas": "Natural grassland and sparsely vegetated areas",
    "Inland marshes": "Inland wetlands",
    "Peat bogs": "Inland wetlands",
    "Salt marshes": "Coastal wetlands",
    "Salines": "Coastal wetlands",
    "Intertidal flats": "Coastal wetlands",
    "Water courses": "Inland waters",
    "Water bodies": "Inland waters",
    "Coastal lagoons": "Marine waters",
    "Estuaries": "Marine waters",
    "Sea and ocean": "Marine waters",
}

# ImageNet normalization parameters for Optical RGB
OPTICAL_MEAN = [0.485, 0.456, 0.406]
OPTICAL_STD = [0.229, 0.224, 0.225]


def load_sar_patch(sar_path: str, target_size=(128, 128)) -> torch.Tensor:
    """
    Loads a SAR GeoTIFF or image, converts to dB scale if stored as linear backscatter,
    and normalizes into standard [0, 1] range.
    """
    # Open with PIL (handles single-band GeoTIFF / PNG)
    sar_img = Image.open(sar_path)
    sar_arr = np.array(sar_img, dtype=np.float32)

    # Convert linear backscatter intensity to logarithmic dB scale if needed
    if np.min(sar_arr) >= 0:
        # Values are non-negative linear power -> convert to dB: 10 * log10(max(val, 1e-7))
        sar_db = 10.0 * np.log10(np.maximum(sar_arr, 1e-7))
    else:
        # Already provided in dB scale
        sar_db = sar_arr

    # Standard SAR dB clipping: backscatter rarely falls outside [-35 dB, +5 dB]
    sar_clipped = np.clip(sar_db, -35.0, 5.0)
    # Min-max scale dB range [-35, 5] -> [0.0, 1.0]
    sar_norm = (sar_clipped - (-35.0)) / (5.0 - (-35.0))

    # Convert to tensor (1, H, W) and resize to target_size
    sar_tensor = torch.from_numpy(sar_norm).unsqueeze(0)  # (1, H, W)
    if sar_tensor.shape[1:] != target_size:
        sar_tensor = TF.resize(sar_tensor, target_size, interpolation=TF.InterpolationMode.BILINEAR)

    return sar_tensor


def load_optical_patch(opt_path: str, target_size=(128, 128)) -> torch.Tensor:
    """
    Loads optical RGB image, resizes to target_size, and applies ImageNet normalization.
    Supports:
    1) Directory path containing S2 bands (B04, B03, B02 GeoTIFFs or _rgb.png)
    2) Direct path to B04 GeoTIFF (inferring B03 and B02)
    3) Standard RGB image file (PNG/JPG)
    """
    if os.path.isdir(opt_path):
        patch_id = os.path.basename(opt_path.rstrip("/\\"))
        rgb_png = os.path.join(opt_path, f"{patch_id}_rgb.png")
        b04_p = os.path.join(opt_path, f"{patch_id}_B04.tif")
        if os.path.exists(rgb_png):
            opt_img = Image.open(rgb_png).convert("RGB")
            opt_img = opt_img.resize(target_size, Image.BILINEAR)
            opt_tensor = TF.to_tensor(opt_img)
        elif os.path.exists(b04_p):
            b03_p = os.path.join(opt_path, f"{patch_id}_B03.tif")
            b02_p = os.path.join(opt_path, f"{patch_id}_B02.tif")
            b04 = np.array(Image.open(b04_p), dtype=np.float32)
            b03 = np.array(Image.open(b03_p), dtype=np.float32)
            b02 = np.array(Image.open(b02_p), dtype=np.float32)
            # Clip surface reflectance to [0, 3000] and scale to [0, 1]
            b04 = np.clip(b04, 0.0, 3000.0) / 3000.0
            b03 = np.clip(b03, 0.0, 3000.0) / 3000.0
            b02 = np.clip(b02, 0.0, 3000.0) / 3000.0
            rgb_arr = np.stack([b04, b03, b02], axis=0)  # (3, H, W)
            opt_tensor = torch.from_numpy(rgb_arr)
            if opt_tensor.shape[1:] != target_size:
                opt_tensor = TF.resize(opt_tensor, target_size, interpolation=TF.InterpolationMode.BILINEAR)
        else:
            raise FileNotFoundError(f"No valid optical bands found in {opt_path}")
    elif opt_path.endswith("_B04.tif"):
        b03_p = opt_path.replace("_B04.tif", "_B03.tif")
        b02_p = opt_path.replace("_B04.tif", "_B02.tif")
        b04 = np.array(Image.open(opt_path), dtype=np.float32)
        b03 = np.array(Image.open(b03_p), dtype=np.float32)
        b02 = np.array(Image.open(b02_p), dtype=np.float32)
        b04 = np.clip(b04, 0.0, 3000.0) / 3000.0
        b03 = np.clip(b03, 0.0, 3000.0) / 3000.0
        b02 = np.clip(b02, 0.0, 3000.0) / 3000.0
        rgb_arr = np.stack([b04, b03, b02], axis=0)
        opt_tensor = torch.from_numpy(rgb_arr)
        if opt_tensor.shape[1:] != target_size:
            opt_tensor = TF.resize(opt_tensor, target_size, interpolation=TF.InterpolationMode.BILINEAR)
    else:
        opt_img = Image.open(opt_path).convert("RGB")
        opt_img = opt_img.resize(target_size, Image.BILINEAR)
        opt_tensor = TF.to_tensor(opt_img)

    # Standard per-band ImageNet normalization
    opt_tensor = TF.normalize(opt_tensor, mean=OPTICAL_MEAN, std=OPTICAL_STD)
    return opt_tensor


def map_raw_labels_to_19(raw_labels: list) -> list:
    """Maps raw 43-class or 19-class labels to consolidated 19-class taxonomy."""
    mapped = set()
    for lbl in raw_labels:
        if lbl in BIGEARTHNET_19_CLASSES:
            mapped.add(lbl)
        elif lbl in BIGEARTHNET_43_TO_19:
            mapped.add(BIGEARTHNET_43_TO_19[lbl])
    return list(mapped)


class BenGeDataset(Dataset):
    """
    Paired Sentinel-1 SAR and Sentinel-2 Optical Dataset for multi-label classification.
    """

    def __init__(
        self,
        data_dir="/home/pavitra/satquery/data/ben-ge-8k",
        classes=None,
        target_size=(128, 128),
        max_samples=None,
    ):
        self.data_dir = data_dir
        self.classes = classes if classes is not None else BIGEARTHNET_19_CLASSES
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.target_size = target_size
        self.samples = []

        meta_csv = os.path.join(data_dir, "ben-ge-8k_meta.csv")
        s2_dir = os.path.join(data_dir, "sentinel-2")
        s1_dir = os.path.join(data_dir, "sentinel-1")

        if os.path.exists(meta_csv):
            with open(meta_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    s2_id = row["patch_id"]
                    s1_id = row["patch_id_s1"]
                    opt_folder = os.path.join(s2_dir, s2_id)
                    sar_file = os.path.join(s1_dir, s1_id, f"{s1_id}_VV.tif")
                    meta_file = os.path.join(s2_dir, s2_id, f"{s2_id}_labels_metadata.json")

                    if os.path.exists(opt_folder) and os.path.exists(sar_file) and os.path.exists(meta_file):
                        self.samples.append({
                            "opt_path": opt_folder,
                            "sar_path": sar_file,
                            "meta_path": meta_file,
                            "patch_id": s2_id,
                            "patch_id_s1": s1_id,
                        })
        else:
            # Fallback folder scan if CSV not present
            if os.path.exists(s1_dir):
                s1_folders = sorted([d for d in os.listdir(s1_dir) if os.path.isdir(os.path.join(s1_dir, d))])
                for s1_name in s1_folders:
                    s1_json_candidates = glob.glob(os.path.join(s1_dir, s1_name, "*_labels_metadata.json"))
                    if not s1_json_candidates:
                        continue
                    try:
                        with open(s1_json_candidates[0], "r", encoding="utf-8") as f:
                            s1_meta = json.load(f)
                        s2_name = s1_meta.get("corresponding_s2_patch")
                        if not s2_name:
                            continue

                        opt_folder = os.path.join(s2_dir, s2_name)
                        sar_file = os.path.join(s1_dir, s1_name, f"{s1_name}_VV.tif")
                        s2_meta_json = os.path.join(s2_dir, s2_name, f"{s2_name}_labels_metadata.json")

                        if os.path.exists(opt_folder) and os.path.exists(sar_file) and os.path.exists(s2_meta_json):
                            self.samples.append({
                                "opt_path": opt_folder,
                                "sar_path": sar_file,
                                "meta_path": s2_meta_json,
                                "patch_id": s2_name,
                                "patch_id_s1": s1_name,
                            })
                    except Exception:
                        continue

        if max_samples is not None and max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 1. Optical RGB (3 channels)
        opt_tensor = load_optical_patch(item["opt_path"], target_size=self.target_size)

        # 2. SAR (1 channel)
        if item["sar_path"] and os.path.exists(item["sar_path"]):
            sar_tensor = load_sar_patch(item["sar_path"], target_size=self.target_size)
        else:
            # If SAR file not yet unpacked, generate a structured single-channel mock from optical luminance
            # to permit pipeline testing without halting execution
            sar_pseudo = (0.299 * opt_tensor[0] + 0.587 * opt_tensor[1] + 0.114 * opt_tensor[2]).unsqueeze(0)
            sar_tensor = sar_pseudo

        # 3. Multi-hot ground truth label
        with open(item["meta_path"]) as f:
            raw_labels = json.load(f).get("labels", [])

        mapped_labels = map_raw_labels_to_19(raw_labels)
        target = torch.zeros(len(self.classes), dtype=torch.float32)
        for lbl in mapped_labels:
            if lbl in self.class_to_idx:
                target[self.class_to_idx[lbl]] = 1.0

        return {
            "sar": sar_tensor,
            "optical": opt_tensor,
            "target": target,
            "labels": mapped_labels,
            "patch_id": item["patch_id"],
        }


def fusion_collate_fn(batch):
    """Custom collator handling variable length label lists without DataLoader errors."""
    return {
        "sar": torch.stack([b["sar"] for b in batch], dim=0),
        "optical": torch.stack([b["optical"] for b in batch], dim=0),
        "target": torch.stack([b["target"] for b in batch], dim=0),
        "labels": [b["labels"] for b in batch],
        "patch_id": [b["patch_id"] for b in batch],
    }


# ---------- Dataset Smoke Test (Task 3) ----------
if __name__ == "__main__":
    print("=" * 60)
    print("TESTING BEN-GE DATASET LOADER (20 SAMPLES)")
    print("=" * 60)

    dataset = BenGeDataset(max_samples=20)
    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample 0 Inspection:")
        print(f"  Patch ID:       {sample['patch_id']}")
        print(f"  Optical Shape:  {sample['optical'].shape} (dtype: {sample['optical'].dtype})")
        print(f"  SAR Shape:      {sample['sar'].shape} (dtype: {sample['sar'].dtype})")
        print(f"  Target Shape:   {sample['target'].shape} (sum: {sample['target'].sum().item()})")
        print(f"  Active Labels:  {sample['labels']}")

        # Validate across 20 samples
        print(f"\nVerifying batch collation across {min(len(dataset), 20)} samples...")
        for i in range(min(len(dataset), 20)):
            s = dataset[i]
            assert s["optical"].shape == (3, 128, 128)
            assert s["sar"].shape == (1, 128, 128)
            assert s["target"].shape == (len(BIGEARTHNET_19_CLASSES),)

        print("All 20 test samples verified successfully!")
    else:
        print("No samples found yet in data directory.")
