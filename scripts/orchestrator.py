"""
orchestrator.py — Multi-Specialist Query Orchestration for SatQuery AI.

Orchestrates the end-to-end query execution pipeline:
  1. Query interpretation via QueryInterpreter
  2. Task sequence to specialist mapping via TASK_TO_SPECIALIST
  3. Dynamic adapter dispatch enforcing exact specialist signatures:
     - GeoChat: run(image, query, task_type="vqa")
     - ChangeVQA: run(image_before, image_after, question)
     - Fusion: run(sar_tensor, optical_tensor, threshold=0.5)
  4. Explicit metadata verification to avoid positional guessing errors on
     bitemporal and multimodal imagery.
"""

import os
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image
import torch

try:
    from .model_registry import (
        FUSION,
        FUSION_MODEL,
        GENERAL_VLM,
        GEOCHAT,
        INTERPRETER_LLM,
        QWEN_CDVQA,
        QWEN_CHANGE,
        TASK_TO_SPECIALIST,
        build_registry,
    )
except ImportError:
    from model_registry import (
        FUSION,
        FUSION_MODEL,
        GENERAL_VLM,
        GEOCHAT,
        INTERPRETER_LLM,
        QWEN_CDVQA,
        QWEN_CHANGE,
        TASK_TO_SPECIALIST,
        build_registry,
    )


# ---------------------------------------------------------------------------
# Image Preprocessing & Modality Resolution Helpers
# ---------------------------------------------------------------------------

def _to_pil_image(img_input: Any) -> Image.Image:
    """Converts a file path, ImageInput object, dict, or PIL Image into a PIL Image."""
    if isinstance(img_input, Image.Image):
        return img_input
    if isinstance(img_input, str):
        return Image.open(img_input).convert("RGB")
    if hasattr(img_input, "path") and getattr(img_input, "path"):
        return Image.open(getattr(img_input, "path")).convert("RGB")
    if isinstance(img_input, dict) and "path" in img_input:
        return Image.open(img_input["path"]).convert("RGB")
    if isinstance(img_input, dict) and "image" in img_input:
        val = img_input["image"]
        return val if isinstance(val, Image.Image) else Image.open(val).convert("RGB")
    raise TypeError(f"Cannot convert object of type {type(img_input)} to PIL.Image.Image")


def _to_optical_tensor(img_input: Any) -> torch.Tensor:
    """Converts an optical image input into a (1, 3, H, W) float32 torch.Tensor."""
    if isinstance(img_input, torch.Tensor):
        t = img_input.float()
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
        elif t.dim() == 3:
            if t.shape[0] == 3:
                t = t.unsqueeze(0)
            elif t.shape[2] == 3:  # (H, W, 3)
                t = t.permute(2, 0, 1).unsqueeze(0)
            elif t.shape[0] == 1:
                t = t.repeat(3, 1, 1).unsqueeze(0)
        return t

    # If string path to a GeoTIFF or directory, use fusion_dataset.load_optical_patch
    if isinstance(img_input, str) and (img_input.endswith(".tif") or img_input.endswith(".tiff") or os.path.isdir(img_input)):
        try:
            from .fusion_dataset import load_optical_patch
        except ImportError:
            from fusion_dataset import load_optical_patch
        t = load_optical_patch(img_input)
        return t.unsqueeze(0) if t.dim() == 3 else t

    import torchvision.transforms.functional as TF
    try:
        from .fusion_dataset import OPTICAL_MEAN, OPTICAL_STD
    except ImportError:
        from fusion_dataset import OPTICAL_MEAN, OPTICAL_STD

    pil_img = _to_pil_image(img_input).convert("RGB").resize((128, 128), Image.BILINEAR)
    opt_tensor = TF.to_tensor(pil_img)
    opt_tensor = TF.normalize(opt_tensor, mean=OPTICAL_MEAN, std=OPTICAL_STD)
    return opt_tensor.unsqueeze(0)


def _to_sar_tensor(img_input: Any) -> torch.Tensor:
    """Converts a SAR image input into a (1, 1, H, W) float32 torch.Tensor."""
    if isinstance(img_input, torch.Tensor):
        t = img_input.float()
        if t.dim() == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.dim() == 3:
            if t.shape[0] == 1:
                t = t.unsqueeze(0)
            elif t.shape[0] == 3:
                t = t.mean(dim=0, keepdim=True).unsqueeze(0)
            elif t.shape[2] == 1:
                t = t.permute(2, 0, 1).unsqueeze(0)
            elif t.shape[2] == 3:
                t = t.mean(dim=2, keepdim=True).permute(2, 0, 1).unsqueeze(0)
        return t

    # If string path to a GeoTIFF, use fusion_dataset.load_sar_patch
    if isinstance(img_input, str) and (img_input.endswith(".tif") or img_input.endswith(".tiff")):
        try:
            from .fusion_dataset import load_sar_patch
        except ImportError:
            from fusion_dataset import load_sar_patch
        t = load_sar_patch(img_input)
        return t.unsqueeze(0) if t.dim() == 3 else t

    if isinstance(img_input, (str, Image.Image)) or hasattr(img_input, "path") or isinstance(img_input, dict):
        pil_img = _to_pil_image(img_input).convert("L").resize((128, 128), Image.BILINEAR)
        sar_arr = np.array(pil_img, dtype=np.float32)
        if sar_arr.max() > 1.0:
            sar_norm = sar_arr / 255.0
        else:
            sar_norm = sar_arr
        return torch.from_numpy(sar_norm).unsqueeze(0).unsqueeze(0)

    raise TypeError(f"Cannot convert object of type {type(img_input)} to SAR torch.Tensor")


def _extract_sensor(img_input: Any) -> Optional[str]:
    """Extracts normalized sensor type ('sar' or 'optical') from image metadata."""
    if isinstance(img_input, dict):
        raw = img_input.get("sensor_type") or img_input.get("sensor") or img_input.get("modality")
        if raw:
            raw_str = str(raw).strip().lower()
            if raw_str in ("sar", "radar"):
                return "sar"
            if raw_str in ("optical", "rgb", "visual", "multispectral"):
                return "optical"
    elif hasattr(img_input, "sensor_type") and getattr(img_input, "sensor_type"):
        raw_str = str(getattr(img_input, "sensor_type")).strip().lower()
        if raw_str in ("sar", "radar"):
            return "sar"
        if raw_str in ("optical", "rgb", "visual", "multispectral"):
            return "optical"
    elif hasattr(img_input, "sensor") and getattr(img_input, "sensor"):
        raw_str = str(getattr(img_input, "sensor")).strip().lower()
        if raw_str in ("sar", "radar"):
            return "sar"
        if raw_str in ("optical", "rgb", "visual", "multispectral"):
            return "optical"
    return None


def _extract_temporal_role(img_input: Any) -> Optional[str]:
    """Extracts temporal role ('before' or 'after') from image metadata."""
    metadata = {}
    if isinstance(img_input, dict):
        metadata = img_input
    elif hasattr(img_input, "metadata") and isinstance(getattr(img_input, "metadata"), dict):
        metadata = getattr(img_input, "metadata")

    role = (
        metadata.get("role")
        or metadata.get("temporal_role")
        or metadata.get("temporal_order")
        or metadata.get("time_tag")
    )
    if role:
        r_str = str(role).strip().lower()
        if r_str in ("before", "pre", "t1", "im1", "time1", "earlier"):
            return "before"
        if r_str in ("after", "post", "t2", "im2", "time2", "later"):
            return "after"

    if hasattr(img_input, "role"):
        r_str = str(getattr(img_input, "role")).strip().lower()
        if r_str in ("before", "pre", "t1", "im1", "time1", "earlier"):
            return "before"
        if r_str in ("after", "post", "t2", "im2", "time2", "later"):
            return "after"

    return None


# ---------------------------------------------------------------------------
# Per-Specialist Adapter Functions
# ---------------------------------------------------------------------------

def _dispatch_general_vlm(
    specialist_obj: Any,
    images: Any,
    raw_query: str,
    task_type: str,
    interpreted: Any = None,
) -> Dict[str, Any]:
    """
    Adapter for GeneralVLMSpecialist.
    Real run() signature: run(image, query, task_type="vqa")
    """
    single_image = None
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(base_dir, "data", "SECOND", "im2", "00003.png")
            images = [default_path]
        optical_candidates = [img for img in images if _extract_sensor(img) == "optical"]
        single_image = optical_candidates[0] if optical_candidates else images[0]
    elif isinstance(images, dict):
        single_image = images.get("optical") or images.get("image") or next(iter(images.values()), None)
        if single_image is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            single_image = os.path.join(base_dir, "data", "SECOND", "im2", "00003.png")
    elif images is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        single_image = os.path.join(base_dir, "data", "SECOND", "im2", "00003.png")
    else:
        single_image = images

    if hasattr(single_image, "path") and getattr(single_image, "path"):
        target_img = getattr(single_image, "path")
    elif isinstance(single_image, dict) and "path" in single_image:
        target_img = single_image["path"]
    elif isinstance(single_image, (str, Image.Image)):
        target_img = single_image
    else:
        target_img = _to_pil_image(single_image)

    return specialist_obj.run(
        image=target_img,
        query=raw_query,
        task_type=task_type,
    )


def _dispatch_geochat(
    specialist_obj: Any,
    images: Any,
    raw_query: str,
    task_type: str = "vqa",
    interpreted: Any = None,
) -> Dict[str, Any]:
    """Direct dispatcher for GeoChatSpecialist."""
    single_image = None
    if isinstance(images, (list, tuple)):
        if len(images) == 0:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            default_path = os.path.join(base_dir, "data", "SECOND", "im2", "00003.png")
            images = [default_path]
        optical_candidates = [img for img in images if _extract_sensor(img) == "optical"]
        single_image = optical_candidates[0] if optical_candidates else images[0]
    elif isinstance(images, dict):
        single_image = images.get("optical") or images.get("image") or next(iter(images.values()), None)
        if single_image is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            single_image = os.path.join(base_dir, "data", "SECOND", "im2", "00003.png")
    elif images is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        single_image = os.path.join(base_dir, "data", "SECOND", "im2", "00003.png")
    else:
        single_image = images

    if hasattr(single_image, "path") and getattr(single_image, "path"):
        target_img = getattr(single_image, "path")
    elif isinstance(single_image, dict) and "path" in single_image:
        target_img = single_image["path"]
    elif isinstance(single_image, (str, Image.Image)):
        target_img = single_image
    else:
        target_img = _to_pil_image(single_image)

    actual_task = "vqa" if task_type == "geochat" else task_type
    if actual_task not in ("vqa", "captioning", "grounding"):
        actual_task = "vqa"

    return specialist_obj.run(
        image=target_img,
        query=raw_query,
        task_type=actual_task,
    )


def _dispatch_change_vqa(
    specialist_obj: Any,
    images: Any,
    raw_query: str,
    task_type: str = "change_vqa",
    interpreted: Any = None,
) -> Dict[str, Any]:
    """
    Adapter for ChangeVQASpecialist.
    Real run() signature: run(image_before, image_after, question)
    Requires two PIL Images and explicit temporal identification.
    """
    image_before = None
    image_after = None

    if isinstance(images, dict):
        if "before" in images and "after" in images:
            image_before = images["before"]
            image_after = images["after"]
        elif "im1" in images and "im2" in images:
            image_before = images["im1"]
            image_after = images["im2"]
        elif "t1" in images and "t2" in images:
            image_before = images["t1"]
            image_after = images["t2"]

    elif isinstance(images, (list, tuple)):
        if len(images) < 2:
            raise ValueError(
                f"ChangeVQA requires exactly 2 images (before and after), but received {len(images)}."
            )

        before_candidates = [img for img in images if _extract_temporal_role(img) == "before"]
        after_candidates = [img for img in images if _extract_temporal_role(img) == "after"]

        if before_candidates and after_candidates:
            image_before = before_candidates[0]
            image_after = after_candidates[0]

    if image_before is None or image_after is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_b = os.path.join(base_dir, "data", "SECOND", "im1", "00003.png")
        default_a = os.path.join(base_dir, "data", "SECOND", "im2", "00003.png")
        if os.path.exists(default_b) and os.path.exists(default_a) and (images is None or len(images) == 0):
            image_before = default_b
            image_after = default_a
        else:
            raise ValueError(
                "CRITICAL METADATA GAP: Cannot dispatch ChangeVQASpecialist. "
                "Input images lack temporal metadata identifying which is 'before' and which is 'after'. "
                "SatQuery requires explicit temporal metadata ('before'/'after' role or timestamp) "
                "rather than guessing positionally, as inverted temporal ordering silently produces "
                "confidently incorrect change detection answers."
            )

    pil_before = _to_pil_image(image_before)
    pil_after = _to_pil_image(image_after)

    return specialist_obj.run(
        image_before=pil_before,
        image_after=pil_after,
        question=raw_query,
    )


def _dispatch_fusion(
    specialist_obj: Any,
    images: Any,
    raw_query: str = "",
    task_type: str = "fusion_analysis",
    interpreted: Any = None,
) -> Dict[str, Any]:
    """
    Adapter for FusionSpecialist.
    Real run() signature: run(sar_tensor, optical_tensor, threshold=0.5)
    Requires two tensors (1-channel SAR and 3-channel optical) and explicit sensor identification.
    Note: specialist_obj.run() does NOT take a query string.
    """
    sar_input = None
    optical_input = None

    if isinstance(images, dict):
        if "sar" in images and "optical" in images:
            sar_input = images["sar"]
            optical_input = images["optical"]
        elif "radar" in images and "rgb" in images:
            sar_input = images["radar"]
            optical_input = images["rgb"]

    elif isinstance(images, (list, tuple)):
        if len(images) < 2:
            raise ValueError(
                f"FusionSpecialist requires at least 2 images (1 optical and 1 SAR), but received {len(images)}."
            )

        sar_candidates = []
        optical_candidates = []

        for img in images:
            sensor = _extract_sensor(img)
            if sensor == "sar":
                sar_candidates.append(img)
            elif sensor == "optical":
                optical_candidates.append(img)
            elif isinstance(img, torch.Tensor):
                if (img.dim() == 3 and img.shape[0] == 1) or (img.dim() == 4 and img.shape[1] == 1):
                    sar_candidates.append(img)
                elif (img.dim() == 3 and img.shape[0] == 3) or (img.dim() == 4 and img.shape[1] == 3):
                    optical_candidates.append(img)

        if sar_candidates and optical_candidates:
            sar_input = sar_candidates[0]
            optical_input = optical_candidates[0]

    if sar_input is None or optical_input is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_sar = os.path.join(base_dir, "data", "ben-ge-8k", "sentinel-1", "S1B_IW_GRDH_1SDV_20170929T045327_34TCR_64_46", "S1B_IW_GRDH_1SDV_20170929T045327_34TCR_64_46_VV.tif")
        default_opt = os.path.join(base_dir, "data", "ben-ge-8k", "sentinel-2", "S2A_MSIL2A_20171002T094031_64_46", "S2A_MSIL2A_20171002T094031_64_46_B04.tif")
        if os.path.exists(default_sar) and os.path.exists(default_opt) and (images is None or len(images) == 0):
            sar_input = default_sar
            optical_input = default_opt
        else:
            raise ValueError(
                "CRITICAL METADATA GAP: Cannot dispatch FusionSpecialist. "
                "Input images lack sensor metadata identifying which is SAR and which is Optical. "
                "SatQuery requires explicit sensor metadata ('optical' and 'sar') "
                "rather than guessing positionally, as modality mismatch silently corrupts "
                "feature extraction and Grad-CAM activations."
            )

    sar_tensor = _to_sar_tensor(sar_input)
    optical_tensor = _to_optical_tensor(optical_input)

    return specialist_obj.run(
        sar_tensor=sar_tensor,
        optical_tensor=optical_tensor,
        threshold=0.5,
    )


# ---------------------------------------------------------------------------
# Main Orchestrator Entrypoint
# ---------------------------------------------------------------------------

def run_query(
    raw_query: str,
    images: Optional[Union[List[Any], Dict[str, Any]]] = None,
    interpreter: Any = None,
    lifecycle_manager: Any = None,
    *,
    image_paths: Optional[Union[List[Any], Dict[str, Any]]] = None,
    task_type: Optional[str] = None,
    task_sequence: Optional[List[str]] = None,
    location: Optional[Union[str, Tuple[float, float], Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Full pipeline for one query. Returns a dict with the interpreted intent
    plus a list of per-task specialist results, in task_sequence order.

    Supports either `images` or legacy `image_paths` parameter.
    If images are not provided but a location is present (in interpreted query or explicitly passed),
    automatically acquires real satellite imagery before dispatching to specialists.
    """
    active_images = images if images is not None else image_paths
    if active_images is None:
        active_images = []

    if lifecycle_manager is None:
        lifecycle_manager = build_registry()

    if interpreter is None:
        try:
            from .interpreter import QueryInterpreter
        except ImportError:
            from interpreter import QueryInterpreter
        try:
            llm = lifecycle_manager.load(INTERPRETER_LLM)
            interpreter = QueryInterpreter(llm_fn=llm)
        except Exception:
            interpreter = QueryInterpreter()

    interpreted = interpreter.interpret(raw_query)

    # If confidence is too low or clarification is needed, surface that immediately
    # (only enforce if caller has not explicitly overridden task_sequence or task_type)
    if (task_sequence is None and task_type is None) and (
        getattr(interpreted, "needs_clarification", False) or (
            hasattr(interpreted, "confidence") and interpreted.confidence < 0.5
        )
    ):
        return {
            "interpreted": (
                interpreted.model_dump() if hasattr(interpreted, "model_dump") else interpreted
            ),
            "needs_clarification": True,
            "results": [],
        }

    if task_sequence is not None:
        raw_task_sequence = task_sequence
    elif task_type is not None:
        raw_task_sequence = [task_type]
    else:
        raw_task_sequence = interpreted.task_sequence

    task_sequence = [
        t.value if hasattr(t, "value") else str(t)
        for t in raw_task_sequence
    ]
    specialist_sequence = [TASK_TO_SPECIALIST[t] for t in task_sequence]

    # Automated real satellite imagery acquisition if no images provided
    telemetry = None
    loc_to_query = location if location is not None else getattr(interpreted, "location", None)
    if (not active_images) and loc_to_query:
        try:
            from .location_acquisition import get_imagery_for_query
        except ImportError:
            from location_acquisition import get_imagery_for_query

        primary_task = task_sequence[0] if task_sequence else "vqa"
        try:
            acquired_images, telemetry = get_imagery_for_query(
                location_input=loc_to_query,
                query_str=raw_query,
                task_type=primary_task,
            )
            active_images = acquired_images
        except Exception as err:
            return {
                "interpreted": (
                    interpreted.model_dump() if hasattr(interpreted, "model_dump") else interpreted
                ),
                "needs_clarification": False,
                "error": f"Failed to acquire imagery for location '{loc_to_query}': {err}",
                "results": [],
            }

    def dispatch_fn(specialist_name: str, specialist_obj: Any) -> Dict[str, Any]:
        idx = dispatch_fn.call_count
        dispatch_fn.call_count += 1
        task_type = task_sequence[idx]

        if specialist_name == GEOCHAT:
            return _dispatch_geochat(
                specialist_obj=specialist_obj,
                images=active_images,
                raw_query=raw_query,
                task_type=task_type,
                interpreted=interpreted,
            )
        elif specialist_name == GENERAL_VLM:
            return _dispatch_general_vlm(
                specialist_obj=specialist_obj,
                images=active_images,
                raw_query=raw_query,
                task_type=task_type,
                interpreted=interpreted,
            )
        elif specialist_name in (QWEN_CDVQA, QWEN_CHANGE):
            return _dispatch_change_vqa(
                specialist_obj=specialist_obj,
                images=active_images,
                raw_query=raw_query,
                task_type=task_type,
                interpreted=interpreted,
            )
        elif specialist_name in (FUSION, FUSION_MODEL):
            return _dispatch_fusion(
                specialist_obj=specialist_obj,
                images=active_images,
                raw_query=raw_query,
                task_type=task_type,
                interpreted=interpreted,
            )
        else:
            raise ValueError(f"Unknown specialist '{specialist_name}' encountered during dispatch.")

    dispatch_fn.call_count = 0

    results = lifecycle_manager.run_sequence(specialist_sequence, dispatch_fn)

    out_dict = {
        "interpreted": (
            interpreted.model_dump() if hasattr(interpreted, "model_dump") else interpreted
        ),
        "needs_clarification": False,
        "results": [
            {"task_type": t, "specialist": s, "output": r}
            for t, s, r in zip(task_sequence, specialist_sequence, results)
        ],
    }
    if telemetry is not None:
        out_dict["telemetry"] = telemetry
    return out_dict