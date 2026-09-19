"""
router.py — Task-to-Model Router with safety-check fallbacks for SatQuery AI.

The controller layer sitting between the query interpreter (InterpretedQuery)
and the ModelLifecycleManager. Routes individual tasks to specialist models
using a deterministic lookup table and plain if/else safety checks.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

try:
    from .schema import InterpretedQuery, Sensor, Task
except ImportError:
    try:
        from schema import InterpretedQuery, Sensor, Task
    except ImportError:
        Sensor = None  # type: ignore[assignment, misc]
        Task = None  # type: ignore[assignment, misc]
        InterpretedQuery = None  # type: ignore[assignment, misc]

logger = logging.getLogger("SatQuery.Router")


# ---------------------------------------------------------------------------
# Base Routing Table & Model Registrations
# ---------------------------------------------------------------------------

# task_name -> (model_name, images_needed, extra_fields_needed, expected_sensor)
ROUTING_TABLE: Dict[str, Tuple[str, int, List[str], str]] = {
    "vqa":              ("geochat",      1, ["question_text"],  "optical"),
    "captioning":       ("geochat",      1, [],                 "optical"),
    "grounding":        ("geochat",      1, ["target_object"],  "optical"),
    "change_vqa":       ("qwen_change",  2, ["question_text"],  "optical"),  # date_or_time_reference optional
    "fusion_analysis":  ("fusion_model", 2, [],                 "both"),     # requires exactly one optical + one SAR
}

# Stub for future designated backup specialist models
FALLBACK_MODELS: Dict[str, str] = {}


# ---------------------------------------------------------------------------
# Custom Exceptions (all inheriting from RoutingError)
# ---------------------------------------------------------------------------

class RoutingError(Exception):
    """Base exception for all router-related errors in SatQuery AI."""
    pass


class UnknownTaskError(RoutingError, KeyError):
    """Raised when an unrecognized task name not in ROUTING_TABLE is provided."""
    pass


class InsufficientImagesError(RoutingError):
    """Raised when a task requires more images than were provided."""

    def __init__(
        self,
        task_name: str,
        required: int,
        provided: int,
        message: Optional[str] = None,
    ) -> None:
        self.task_name = task_name
        self.required = required
        self.provided = provided
        msg = message or (
            f"Task '{task_name}' requires {required} image(s), "
            f"but {provided} provided."
        )
        super().__init__(msg)


class UnsupportedSensorError(RoutingError):
    """Raised when the sensor type is unsupported for the requested task."""
    pass


class InvalidFusionInputError(RoutingError):
    """Raised when fusion_analysis is requested without a valid optical + SAR pair."""
    pass


class ModelUnavailableError(RoutingError):
    """Raised when a target model cannot be loaded and no fallback is available."""
    pass


# ---------------------------------------------------------------------------
# Image Input & Routing Decision Data Structures
# ---------------------------------------------------------------------------

@dataclass
class ImageInput:
    """
    Image input representation for SatQuery routing.

    Assumption: Since no standalone image container was pre-defined in schema.py,
    this dataclass captures the minimal required image attributes for routing:
    sensor_type ('optical' or 'sar'), optional file path/URI, and optional metadata.
    """
    sensor_type: Literal["optical", "sar"] = "optical"
    path: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RoutingDecision:
    """
    Output contract of the SatQuery Task-to-Model Router.

    Attributes:
        model_name: Specialist model identifier to dispatch (e.g. 'geochat').
        images_used: The subset of input images routed to this model.
        extra_fields: Dictionary of extra fields/parameters required by the task.
        rerouted: True if safety checks rerouted the task from its default model.
        reroute_reason: Explanation if rerouted is True, otherwise None.
        task_name: The task name associated with this decision (e.g. 'vqa').
    """
    model_name: str
    images_used: List[Any]
    extra_fields: Dict[str, Any]
    rerouted: bool = False
    reroute_reason: Optional[str] = None
    task_name: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

def _extract_sensor_type(image: Any) -> str:
    """
    Extracts and normalizes the sensor type from an image representation.
    Supports ImageInput dataclass, objects with .sensor_type / .sensor attributes,
    dictionaries with 'sensor_type' / 'sensor' keys, Sensor enum instances, or strings.
    """
    raw: Any = None
    if isinstance(image, str):
        raw = image
    elif isinstance(image, dict):
        raw = image.get("sensor_type") or image.get("sensor") or image.get("modality")
    elif hasattr(image, "sensor_type"):
        raw = getattr(image, "sensor_type")
    elif hasattr(image, "sensor"):
        raw = getattr(image, "sensor")

    if raw is None:
        return "optical"

    if hasattr(raw, "value"):
        raw = raw.value

    raw_str = str(raw).strip().lower()
    if raw_str in ("sar", "radar"):
        return "sar"
    if raw_str in ("optical", "rgb", "visual", "multispectral"):
        return "optical"
    return raw_str


def _is_fusion_available(model_manager: Optional[Any]) -> bool:
    """
    Checks whether 'fusion_model' is registered and loadable via model_manager.
    """
    if model_manager is None:
        return True

    if hasattr(model_manager, "is_available") and callable(model_manager.is_available):
        return bool(model_manager.is_available("fusion_model"))
    if hasattr(model_manager, "is_registered") and callable(model_manager.is_registered):
        return bool(model_manager.is_registered("fusion_model"))
    if hasattr(model_manager, "has_model") and callable(model_manager.has_model):
        return bool(model_manager.has_model("fusion_model"))
    if hasattr(model_manager, "registered_models"):
        return "fusion_model" in model_manager.registered_models
    if hasattr(model_manager, "models"):
        return "fusion_model" in model_manager.models
    if hasattr(model_manager, "registry"):
        return "fusion_model" in model_manager.registry

    # If only load() is available, test calling load()
    if hasattr(model_manager, "load") and callable(model_manager.load):
        try:
            model_manager.load("fusion_model")
            return True
        except Exception:
            return False

    return True


def _extract_extra_fields(
    extra_field_values: Optional[Union[Dict[str, Any], Any]],
    extra_fields_needed: List[str],
) -> Dict[str, Any]:
    """
    Extracts and normalizes required and optional extra fields from dict or InterpretedQuery.
    """
    extracted: Dict[str, Any] = {}
    if extra_field_values is None:
        return extracted

    # If passed as an InterpretedQuery or object with attributes
    if not isinstance(extra_field_values, dict):
        if hasattr(extra_field_values, "model_dump") and callable(extra_field_values.model_dump):
            source_dict = extra_field_values.model_dump()
        elif hasattr(extra_field_values, "__dict__"):
            source_dict = dict(extra_field_values.__dict__)
        else:
            source_dict = {}

        # Handle raw_query -> question_text alias
        if "raw_query" in source_dict and "question_text" not in source_dict:
            source_dict["question_text"] = source_dict["raw_query"]
    else:
        source_dict = dict(extra_field_values)
        if "raw_query" in source_dict and "question_text" not in source_dict:
            source_dict["question_text"] = source_dict["raw_query"]
        elif "query" in source_dict and "question_text" not in source_dict:
            source_dict["question_text"] = source_dict["query"]

    # Populate all provided fields from source_dict
    for k, v in source_dict.items():
        if v is not None:
            extracted[k] = v

    return extracted


# ---------------------------------------------------------------------------
# Core Routing Logic
# ---------------------------------------------------------------------------

def route_task(
    task_name: Union[str, Any],
    available_images: List[Any],
    extra_field_values: Optional[Union[Dict[str, Any], Any]] = None,
    model_manager: Optional[Any] = None,
) -> RoutingDecision:
    """
    Routes a single task to the appropriate specialist model.

    Evaluates safety checks in order before falling through to the base routing table:
      1. Image count check: Ensure sufficient images exist for the task.
      2. Sensor type check: If a single-sensor task received SAR instead of optical,
         check if fusion_model is available to handle SAR; otherwise raise UnsupportedSensorError.
      3. Fusion input check: fusion_analysis requires at least 2 images with at least
         one optical and one SAR image.
      4. Model load check: Verify model loading via ModelLifecycleManager; fallback if
         registered, otherwise raise ModelUnavailableError.

    Args:
        task_name: Name of the task (str or Task enum).
        available_images: List of available input images (ImageInput, dicts, or objects).
        extra_field_values: Optional dict or InterpretedQuery containing task arguments.
        model_manager: Optional ModelLifecycleManager instance.

    Returns:
        RoutingDecision specifying model_name, images_used, extra_fields, and reroute metadata.
    """
    # Normalize task_name to string key
    task_key = task_name.value if hasattr(task_name, "value") else str(task_name)
    if task_key not in ROUTING_TABLE:
        raise UnknownTaskError(
            f"Unknown task '{task_key}'. Recognized tasks: {list(ROUTING_TABLE.keys())}"
        )

    default_model, images_needed, extra_fields_needed, expected_sensor = ROUTING_TABLE[task_key]
    images = list(available_images) if available_images is not None else []
    resolved_extra_fields = _extract_extra_fields(extra_field_values, extra_fields_needed)

    chosen_model = default_model
    images_used: List[Any] = []
    rerouted = False
    reroute_reason: Optional[str] = None

    # -----------------------------------------------------------------------
    # Safety Check 1 & 3: Fusion Task Validation
    # -----------------------------------------------------------------------
    if task_key == "fusion_analysis":
        if len(images) < 2:
            raise InvalidFusionInputError(
                f"Task 'fusion_analysis' requires at least 2 images (1 optical + 1 SAR), "
                f"but received {len(images)}."
            )

        optical_imgs = [img for img in images if _extract_sensor_type(img) == "optical"]
        sar_imgs = [img for img in images if _extract_sensor_type(img) == "sar"]

        if not optical_imgs or not sar_imgs:
            raise InvalidFusionInputError(
                f"Task 'fusion_analysis' requires exactly one optical and one SAR image, "
                f"but received {len(optical_imgs)} optical and {len(sar_imgs)} SAR image(s)."
            )

        images_used = [optical_imgs[0], sar_imgs[0]]
        chosen_model = default_model

    else:
        # -------------------------------------------------------------------
        # Safety Check 1: Image Count Mismatch (Non-fusion tasks)
        # -------------------------------------------------------------------
        if len(images) < images_needed:
            raise InsufficientImagesError(
                task_name=task_key,
                required=images_needed,
                provided=len(images),
            )

        optical_imgs = [img for img in images if _extract_sensor_type(img) == "optical"]
        sar_imgs = [img for img in images if _extract_sensor_type(img) == "sar"]

        # -------------------------------------------------------------------
        # Safety Check 2: Sensor Type Check for Single-Sensor Tasks
        # -------------------------------------------------------------------
        if expected_sensor == "optical":
            if optical_imgs:
                # Normal condition: Sufficient optical imagery is available
                if len(optical_imgs) < images_needed:
                    raise InsufficientImagesError(
                        task_name=task_key,
                        required=images_needed,
                        provided=len(optical_imgs),
                        message=(
                            f"Task '{task_key}' requires {images_needed} optical image(s), "
                            f"but only {len(optical_imgs)} optical image(s) provided."
                        ),
                    )
                images_used = optical_imgs[:images_needed]
                chosen_model = default_model
            else:
                # Sensor mismatch: Only SAR images are available
                if images_needed == 1:
                    # Single-sensor task (vqa, captioning, grounding)
                    if _is_fusion_available(model_manager):
                        chosen_model = "fusion_model"
                        images_used = sar_imgs[:1]
                        rerouted = True
                        reroute_reason = (
                            f"Single-sensor task '{task_key}' expects optical input but only "
                            f"SAR image was provided; rerouted to 'fusion_model' as fallback"
                        )
                    else:
                        raise UnsupportedSensorError(
                            f"Task '{task_key}' requires optical sensor imagery, but only SAR imagery "
                            f"was provided, and 'fusion_model' is not registered/available."
                        )
                else:
                    raise UnsupportedSensorError(
                        f"Task '{task_key}' requires optical sensor imagery, "
                        f"but only SAR imagery was provided."
                    )

    # -----------------------------------------------------------------------
    # Safety Check 4: Model Load Failure & Fallback Handling
    # -----------------------------------------------------------------------
    if model_manager is not None and hasattr(model_manager, "load") and callable(model_manager.load):
        try:
            model_manager.load(chosen_model)
        except Exception as err:
            logger.error(f"Failed to load model '{chosen_model}': {err}")
            backup = FALLBACK_MODELS.get(chosen_model)
            if backup:
                try:
                    model_manager.load(backup)
                    chosen_model = backup
                    rerouted = True
                    reroute_reason = f"Primary model failed to load; fell back to {backup}"
                except Exception as backup_err:
                    raise ModelUnavailableError(
                        f"Target model '{chosen_model}' and backup model '{backup}' "
                        f"both failed to load: {backup_err}"
                    ) from backup_err
            else:
                raise ModelUnavailableError(
                    f"Target model '{chosen_model}' failed to load and no fallback "
                    f"model is registered: {err}"
                ) from err

    return RoutingDecision(
        model_name=chosen_model,
        images_used=images_used,
        extra_fields=resolved_extra_fields,
        rerouted=rerouted,
        reroute_reason=reroute_reason,
        task_name=task_key,
    )


def route_sequence(
    task_sequence: List[Union[str, Any]],
    available_images: List[Any],
    extra_field_values: Optional[Union[Dict[str, Any], Any]] = None,
    model_manager: Optional[Any] = None,
) -> List[RoutingDecision]:
    """
    Routes an ordered sequence of tasks (e.g. from InterpretedQuery.task_sequence)
    by calling route_task once per task in order.

    Args:
        task_sequence: Ordered list of task names or Task enum values.
        available_images: List of available images with sensor metadata.
        extra_field_values: Dict or InterpretedQuery containing task arguments.
        model_manager: Optional ModelLifecycleManager instance.

    Returns:
        List of RoutingDecision objects in the same order as task_sequence.
    """
    decisions: List[RoutingDecision] = []
    for task in task_sequence:
        decision = route_task(
            task_name=task,
            available_images=available_images,
            extra_field_values=extra_field_values,
            model_manager=model_manager,
        )
        decisions.append(decision)
    return decisions
