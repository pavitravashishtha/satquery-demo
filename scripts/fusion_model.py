import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Real land-cover taxonomy from ben-ge-8k / BigEarthNet (19 classes)
CLASS_NAMES = [
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


class SAREncoder(nn.Module):
    """Small CNN encoder for single-channel SAR input (e.g. VV or VH backscatter)."""

    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        # 3 conv layers: 32 -> 64 -> 128 channels, stride 2 each
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, feature_dim)
        
        self.last_activation = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Last conv layer activation stored for potential Grad-CAM
        self.last_activation = F.relu(self.bn3(self.conv3(x)))
        
        pooled = self.pool(self.last_activation).flatten(1)
        feat = self.fc(pooled)
        return feat


class OpticalEncoder(nn.Module):
    """Small CNN encoder for 3-channel optical RGB input."""

    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        # 3 conv layers: 32 -> 64 -> 128 channels, stride 2 each
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, feature_dim)
        
        self.last_activation = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Last conv layer activation stored for Grad-CAM
        self.last_activation = F.relu(self.bn3(self.conv3(x)))
        
        pooled = self.pool(self.last_activation).flatten(1)
        feat = self.fc(pooled)
        return feat


class FusionModel(nn.Module):
    """Dual-encoder fusion model concatenating SAR and Optical features into an MLP head."""

    def __init__(self, num_classes=len(CLASS_NAMES), feature_dim=256):
        super().__init__()
        self.sar_encoder = SAREncoder(in_channels=1, feature_dim=feature_dim)
        self.optical_encoder = OpticalEncoder(in_channels=3, feature_dim=feature_dim)
        
        # MLP head: 512 (256+256) -> 256 -> num_classes
        self.head = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, sar, optical):
        sar_feat = self.sar_encoder(sar)
        opt_feat = self.optical_encoder(optical)
        fused = torch.cat([sar_feat, opt_feat], dim=1)
        logits = self.head(fused)
        return logits


class GradCAM:
    """Computes Grad-CAM heatmaps from OpticalEncoder's final conv layer."""

    def __init__(self, model: FusionModel):
        self.model = model
        self.gradient = None
        self.hook_handle = None
        self._register_hook()

    def _register_hook(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        target_layer = self.model.optical_encoder.conv3
        self.hook_handle = target_layer.register_full_backward_hook(backward_hook)

    def generate(self, sar_tensor: torch.Tensor, opt_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad()
        
        # Forward pass
        logits = self.model(sar_tensor, opt_tensor)
        target_score = logits[0, class_idx]
        
        # Backward pass for the target class logit
        target_score.backward(retain_graph=True)
        
        # Retrieve gradients and activations
        grads = self.gradient  # (1, 128, H_conv, W_conv)
        activations = self.model.optical_encoder.last_activation  # (1, 128, H_conv, W_conv)
        
        if grads is None or activations is None:
            return np.zeros((opt_tensor.shape[2], opt_tensor.shape[3]), dtype=np.float32)

        # Global average pool the gradients across spatial dims as channel weights
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)  # (1, 128, 1, 1)
        
        # Weighted sum of activation maps
        cam = torch.sum(weights * activations, dim=1, keepdim=True)  # (1, 1, H_conv, W_conv)
        cam = F.relu(cam)
        
        # Interpolate heatmap up to original input size
        cam = F.interpolate(cam, size=(opt_tensor.shape[2], opt_tensor.shape[3]), mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()
        
        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam

    def remove_hook(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


def generate_answer(probs_dict: dict, threshold: float = 0.5) -> str:
    """Generates a natural sentence from predicted class probabilities."""
    detected = [
        f"{cls_name.replace('_', ' ')} (confidence {prob * 100:.1f}%)"
        for cls_name, prob in probs_dict.items()
        if prob >= threshold
    ]
    
    if detected:
        return f"Combining optical and SAR data, the following are present: {', '.join(detected)}."

    # Adaptive candidate reporting for sub-threshold classes (surfacing top signals together)
    sorted_classes = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
    top_candidates = [
        f"{name.replace('_', ' ')} ({prob * 100:.1f}%)"
        for name, prob in sorted_classes[:3]
        if prob >= 0.02
    ]
    if len(top_candidates) >= 2:
        top_str = f"{top_candidates[0]} and {top_candidates[1]}"
        water_related = any("water" in c.lower() or "wetland" in c.lower() for c in top_candidates[:2])
        context_note = ", both consistent with surface moisture and standing water" if water_related else ""
        return f"Combining optical and SAR data, all classes fell below the 50% activation threshold; strongest signals were {top_str}{context_note}."
    elif top_candidates:
        return f"Combining optical and SAR data, all classes fell below the 50% activation threshold; strongest candidate signal was {top_candidates[0]}."
    return "Combining optical and SAR data, no significant land-cover features detected."


class FusionSpecialist:
    """SpecialistModel interface wrapper for the Optical-SAR Fusion model."""

    def __init__(
        self,
        checkpoint_path: str = "/home/pavitra/satquery/checkpoints/fusion_model.pt",
        class_names: list = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.checkpoint_path = checkpoint_path
        self.class_names = class_names if class_names is not None else CLASS_NAMES
        self.device = device
        self.model = None
        self.grad_cam = None

    def load(self):
        """Loads model into GPU memory."""
        self.model = FusionModel(num_classes=len(self.class_names))
        if os.path.exists(self.checkpoint_path):
            state_dict = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            print(f"[FusionSpecialist] Loaded checkpoint from: {self.checkpoint_path}")
        else:
            print(f"[FusionSpecialist] No checkpoint found at {self.checkpoint_path}; using initialized weights.")
            
        self.model.to(self.device)
        self.model.eval()
        self.grad_cam = GradCAM(self.model)
        print(f"[FusionSpecialist] Model loaded into {self.device}. Memory: {self.memory_footprint_gb()} GB")

    def run(self, sar_tensor: torch.Tensor, optical_tensor: torch.Tensor, threshold: float = 0.5):
        """
        Runs inference and Grad-CAM on paired SAR and optical tensors.
        sar_tensor: (1, 1, H, W) or (1, H, W)
        optical_tensor: (1, 3, H, W) or (3, H, W)
        Returns:
            {
                "answer": str,
                "confidence": float,
                "task_type": "optical_sar_fusion",
                "evidence_maps": {class_name: np.ndarray},
                "top_classes": List[Tuple[str, float]],
            }
        """
        if sar_tensor.dim() == 3:
            sar_tensor = sar_tensor.unsqueeze(0)
        if optical_tensor.dim() == 3:
            optical_tensor = optical_tensor.unsqueeze(0)

        sar_tensor = sar_tensor.to(self.device)
        optical_tensor = optical_tensor.to(self.device)

        # Forward pass with gradients enabled for Grad-CAM
        with torch.enable_grad():
            logits = self.model(sar_tensor, optical_tensor)
            probs = torch.sigmoid(logits).squeeze(0)  # (num_classes,)
            probs_dict = {
                name: probs[i].item() for i, name in enumerate(self.class_names)
            }

            # Generate Grad-CAM heatmaps for detected classes or primary candidate (>= 0.05)
            evidence_maps = {}
            active_indices = [i for i in range(len(self.class_names)) if probs[i].item() >= threshold]
            if not active_indices:
                top_i = int(torch.argmax(probs).item())
                if probs[top_i].item() >= 0.05:
                    active_indices = [top_i]

            for i in active_indices:
                heatmap = self.grad_cam.generate(sar_tensor, optical_tensor, class_idx=i)
                evidence_maps[self.class_names[i]] = heatmap

        answer = generate_answer(probs_dict, threshold=threshold)

        # Overall confidence: average probability among classes that cleared threshold (or max prob if none)
        active_probs = [p for p in probs_dict.values() if p >= threshold]
        overall_conf = float(np.mean(active_probs)) if active_probs else float(np.max(list(probs_dict.values())))

        sorted_classes = sorted(probs_dict.items(), key=lambda x: x[1], reverse=True)
        top_classes = [(name, float(prob)) for name, prob in sorted_classes[:3]]

        return {
            "answer": answer,
            "confidence": round(overall_conf, 4),
            "task_type": "optical_sar_fusion",
            "evidence_maps": evidence_maps,
            "top_classes": top_classes,
        }

    def unload(self):
        """Frees GPU memory cleanly."""
        if self.grad_cam is not None:
            self.grad_cam.remove_hook()
            self.grad_cam = None
        del self.model
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def memory_footprint_gb(self):
        """Returns real measured allocated VRAM in GB."""
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            return round(allocated, 4)
        return 0.0


# ---------- Dummy Tensor Smoke Test (Task 1) ----------
if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING ARCHITECTURE VERIFICATION WITH DUMMY TENSORS")
    print("=" * 60)

    specialist = FusionSpecialist()
    specialist.load()

    # Create dummy random tensors simulating 128x128 SAR (1-channel) and Optical (3-channel)
    dummy_sar = torch.randn(1, 1, 128, 128)
    dummy_opt = torch.randn(1, 3, 128, 128)

    print("\nRunning forward pass and Grad-CAM on dummy inputs...")
    result = specialist.run(dummy_sar, dummy_opt, threshold=0.4)

    print(f"Generated Answer:   {result['answer']}")
    print(f"Overall Confidence: {result['confidence']}")
    print(f"Task Type:          {result['task_type']}")
    print(f"Evidence Maps:      {list(result['evidence_maps'].keys())}")
    for name, cam in result['evidence_maps'].items():
        print(f"  - Heatmap '{name}' shape: {cam.shape}, range: [{cam.min():.3f}, {cam.max():.3f}]")

    print(f"Measured VRAM before unload: {specialist.memory_footprint_gb()} GB")
    specialist.unload()
    print(f"Measured VRAM after unload:  {specialist.memory_footprint_gb()} GB")
    print("\nArchitecture smoke test passed successfully!")
