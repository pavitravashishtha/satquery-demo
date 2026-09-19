"""
controller.py — End-to-End Orchestrator for SatQuery AI.

Connects the QueryInterpreter (NLP query analysis), Router (specialist model
selection & safety-check fallbacks), and ModelLifecycleManager (VRAM model swapping).
Dispatches inference calls, tracks execution audits, and synthesizes final answers.
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

try:
    from .interpreter import QueryInterpreter
    from .model_lifecycle_manager import ModelLifecycleManager
    from .router import (
        ImageInput,
        RoutingDecision,
        RoutingError,
        route_sequence,
        route_task,
    )
    from .schema import InterpretedQuery, Sensor, Task
except ImportError:
    from interpreter import QueryInterpreter
    from model_lifecycle_manager import ModelLifecycleManager
    from router import (
        ImageInput,
        RoutingDecision,
        RoutingError,
        route_sequence,
        route_task,
    )
    from schema import InterpretedQuery, Sensor, Task

logger = logging.getLogger("SatQuery.Controller")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class TaskOutput:
    """
    Execution output and audit metadata for a single task specialist call.

    Attributes:
        task_name: Task performed (e.g. 'vqa', 'grounding', 'change_vqa').
        model_name: Specialist model invoked (e.g. 'geochat', 'qwen_change', 'fusion_model').
        images_used: The subset of input images routed to this specialist.
        extra_fields: Task parameters and context passed to the specialist.
        rerouted: True if safety checks rerouted the task from its default model.
        reroute_reason: Explanation if rerouted is True, otherwise None.
        raw_output: The raw return value from the specialist inference call.
        error: Optional error message if execution failed for this specific task.
    """
    task_name: str
    model_name: str
    images_used: List[Any]
    extra_fields: Dict[str, Any]
    rerouted: bool = False
    reroute_reason: Optional[str] = None
    raw_output: Any = None
    error: Optional[str] = None


@dataclass
class QueryResult:
    """
    End-to-end execution result of a SatQuery AI query.

    Attributes:
        success: True if interpretation, routing, and task dispatch completed without fatal error.
        raw_query: Original user query string.
        interpreted: InterpretedQuery object produced by QueryInterpreter (or None on failure).
        task_outputs: List of TaskOutput objects, one per task in task_sequence.
        execution_summary: Human-readable auditable execution summary for judges/operators.
        final_answer: Synthesized response string across all executed tasks.
        error_message: Detailed error message if query execution failed.
    """
    success: bool
    raw_query: str
    interpreted: Optional[InterpretedQuery]
    task_outputs: List[TaskOutput]
    execution_summary: str
    final_answer: str
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# QueryController Orchestrator
# ---------------------------------------------------------------------------

class QueryController:
    """
    Main controller orchestrating QueryInterpreter -> Router -> ModelLifecycleManager.

    Supports pluggable specialist inference functions via `inference_fn(model_name, model_obj, decision)`.
    """

    def __init__(
        self,
        interpreter: Optional[QueryInterpreter] = None,
        model_manager: Optional[ModelLifecycleManager] = None,
        inference_fn: Optional[Callable[[str, Any, RoutingDecision], Any]] = None,
    ) -> None:
        self.interpreter = interpreter or QueryInterpreter(llm_fn=None)
        self.model_manager = model_manager
        self.inference_fn = inference_fn

    def handle_query(
        self,
        raw_query: str,
        images: Optional[List[Any]] = None,
        interpreter: Optional[QueryInterpreter] = None,
        model_manager: Optional[ModelLifecycleManager] = None,
        inference_fn: Optional[Callable[[str, Any, RoutingDecision], Any]] = None,
    ) -> QueryResult:
        """
        Executes a natural-language query end-to-end against available images.

        Flow:
          1. QueryInterpreter.interpret(raw_query) -> InterpretedQuery
          2. router.route_sequence(task_sequence, images, interpreted, model_manager) -> List[RoutingDecision]
          3. For each RoutingDecision:
             - model_manager.load(decision.model_name)
             - inference_fn(model_name, model_obj, decision) -> raw_output
             - collect TaskOutput
          4. Synthesize final_answer and auditable execution_summary.

        Args:
            raw_query: Natural-language query string from user.
            images: List of available input images (ImageInput, dict, or objects).
            interpreter: Optional override QueryInterpreter instance.
            model_manager: Optional override ModelLifecycleManager instance.
            inference_fn: Optional override inference dispatch function.

        Returns:
            QueryResult containing structured execution results, outputs, and audit summary.
        """
        active_interpreter = interpreter or self.interpreter
        active_manager = model_manager if model_manager is not None else self.model_manager
        active_inference_fn = inference_fn if inference_fn is not None else self.inference_fn

        images_list = list(images) if images is not None else []
        clean_query = raw_query.strip() if raw_query else ""

        # -------------------------------------------------------------------
        # Step 1: Interpret query text into structured schema
        # -------------------------------------------------------------------
        try:
            interpreted = active_interpreter.interpret(clean_query)
        except Exception as err:
            logger.error(f"Interpretation failed for query '{clean_query}': {err}")
            err_msg = f"Interpreter failure: {err}"
            return QueryResult(
                success=False,
                raw_query=clean_query,
                interpreted=None,
                task_outputs=[],
                execution_summary=f"=== SatQuery Execution Summary ===\nQuery: \"{clean_query}\"\nStatus: FAILED ({err_msg})",
                final_answer=f"Error: {err_msg}",
                error_message=err_msg,
            )

        # -------------------------------------------------------------------
        # Step 2: Route planned task sequence to specialist models
        # -------------------------------------------------------------------
        try:
            decisions = route_sequence(
                task_sequence=interpreted.task_sequence,
                available_images=images_list,
                extra_field_values=interpreted,
                model_manager=active_manager,
            )
        except RoutingError as err:
            logger.error(f"Routing failed for query '{clean_query}': {err}")
            err_msg = f"Routing failure: {err}"
            summary = self._build_failure_summary(interpreted, err_msg)
            return QueryResult(
                success=False,
                raw_query=clean_query,
                interpreted=interpreted,
                task_outputs=[],
                execution_summary=summary,
                final_answer=f"Error: {err_msg}",
                error_message=err_msg,
            )
        except Exception as err:
            logger.error(f"Unexpected error during routing for query '{clean_query}': {err}")
            err_msg = f"Unexpected routing error: {err}"
            summary = self._build_failure_summary(interpreted, err_msg)
            return QueryResult(
                success=False,
                raw_query=clean_query,
                interpreted=interpreted,
                task_outputs=[],
                execution_summary=summary,
                final_answer=f"Error: {err_msg}",
                error_message=err_msg,
            )

        # -------------------------------------------------------------------
        # Step 3: Execute specialist inference per task
        # -------------------------------------------------------------------
        task_outputs: List[TaskOutput] = []

        for idx, (task, decision) in enumerate(zip(interpreted.task_sequence, decisions)):
            task_key = task.value if hasattr(task, "value") else str(task)

            # TODO: [Dependent Task Chaining]
            # Currently, tasks in a compound query execute independently against the input images.
            # If sequential task dependency is required (e.g., passing Task 1's grounding bounding box
            # or segmentation mask into Task 2's change_vqa as spatial context/crop), inject the
            # previous task output (task_outputs[-1].raw_output) into decision.extra_fields or
            # decision.images_used right here before dispatching inference.

            # Obtain model object from model_manager (already verified/loaded during routing)
            model_obj = None
            if active_manager is not None and hasattr(active_manager, "load") and callable(active_manager.load):
                try:
                    model_obj = active_manager.load(decision.model_name)
                except Exception as load_err:
                    logger.error(f"Failed to load model '{decision.model_name}': {load_err}")
                    err_msg = f"Model load failure for '{decision.model_name}': {load_err}"
                    return QueryResult(
                        success=False,
                        raw_query=clean_query,
                        interpreted=interpreted,
                        task_outputs=task_outputs,
                        execution_summary=self._build_failure_summary(interpreted, err_msg),
                        final_answer=f"Error: {err_msg}",
                        error_message=err_msg,
                    )

            # Dispatch inference via pluggable hook or default dummy handler
            try:
                if active_inference_fn is not None:
                    raw_output = active_inference_fn(decision.model_name, model_obj, decision)
                else:
                    raw_output = f"[DUMMY OUTPUT for {decision.model_name} on task {task_key}]"
            except Exception as inf_err:
                logger.error(
                    f"Inference execution failed on task '{task_key}' with model '{decision.model_name}': {inf_err}"
                )
                err_msg = f"Inference failure on task '{task_key}': {inf_err}"
                task_outputs.append(
                    TaskOutput(
                        task_name=task_key,
                        model_name=decision.model_name,
                        images_used=decision.images_used,
                        extra_fields=decision.extra_fields,
                        rerouted=decision.rerouted,
                        reroute_reason=decision.reroute_reason,
                        raw_output=None,
                        error=str(inf_err),
                    )
                )
                summary = self._build_execution_summary(interpreted, task_outputs, error_note=err_msg)
                return QueryResult(
                    success=False,
                    raw_query=clean_query,
                    interpreted=interpreted,
                    task_outputs=task_outputs,
                    execution_summary=summary,
                    final_answer=f"Error during execution: {err_msg}",
                    error_message=err_msg,
                )

            task_outputs.append(
                TaskOutput(
                    task_name=task_key,
                    model_name=decision.model_name,
                    images_used=decision.images_used,
                    extra_fields=decision.extra_fields,
                    rerouted=decision.rerouted,
                    reroute_reason=decision.reroute_reason,
                    raw_output=raw_output,
                )
            )

        # -------------------------------------------------------------------
        # Step 4: Synthesize final answer & build execution summary
        # -------------------------------------------------------------------
        final_answer = self._synthesize_final_answer(task_outputs)
        execution_summary = self._build_execution_summary(interpreted, task_outputs)

        return QueryResult(
            success=True,
            raw_query=clean_query,
            interpreted=interpreted,
            task_outputs=task_outputs,
            execution_summary=execution_summary,
            final_answer=final_answer,
            error_message=None,
        )

    # -----------------------------------------------------------------------
    # Summary & Synthesis Helpers
    # -----------------------------------------------------------------------

    def _build_execution_summary(
        self,
        interpreted: InterpretedQuery,
        task_outputs: List[TaskOutput],
        error_note: Optional[str] = None,
    ) -> str:
        """Constructs an auditable, human-readable execution summary."""
        domain_val = interpreted.domain.value if hasattr(interpreted.domain, "value") else str(interpreted.domain)
        sensor_val = interpreted.sensor.value if hasattr(interpreted.sensor, "value") else str(interpreted.sensor)
        source_val = interpreted.source.value if hasattr(interpreted.source, "value") else str(interpreted.source)
        task_seq_vals = [t.value if hasattr(t, "value") else str(t) for t in interpreted.task_sequence]

        lines = [
            "=== SatQuery Auditable Execution Summary ===",
            f"Query: \"{interpreted.raw_query}\"",
            f"Domain: {domain_val} | Implied Sensor: {sensor_val} | Confidence: {interpreted.confidence:.2f} ({source_val})",
            f"Planned Task Sequence: {task_seq_vals}",
        ]
        if interpreted.location:
            lines.append(f"Location: {interpreted.location}")
        if interpreted.date_or_time_reference:
            lines.append(f"Temporal Reference: {interpreted.date_or_time_reference}")
        if interpreted.target_object:
            lines.append(f"Target Object: {interpreted.target_object}")

        lines.append("--- Dispatched Tasks ---")
        if not task_outputs:
            lines.append("  (No tasks executed)")
        for idx, out in enumerate(task_outputs, 1):
            reroute_flag = f" [REROUTED: {out.reroute_reason}]" if out.rerouted else ""
            lines.append(f"Step {idx}: Task='{out.task_name}' -> Model='{out.model_name}'{reroute_flag}")
            if out.extra_fields:
                key_params = ", ".join(
                    f"{k}='{v}'" for k, v in out.extra_fields.items() if k not in ("raw_query",)
                )
                if key_params:
                    lines.append(f"  Parameters: {key_params}")
            lines.append(f"  Images Used: {len(out.images_used)}")
            if out.error:
                lines.append(f"  Error: {out.error}")

        if error_note:
            lines.append(f"Status: FAILED ({error_note})")
        else:
            lines.append("Status: SUCCESS")

        return "\n".join(lines)

    def _build_failure_summary(
        self,
        interpreted: Optional[InterpretedQuery],
        error_msg: str,
    ) -> str:
        """Constructs a minimal failure summary when routing or execution fails early."""
        if interpreted is None:
            return f"=== SatQuery Execution Summary ===\nStatus: FAILED ({error_msg})"

        domain_val = interpreted.domain.value if hasattr(interpreted.domain, "value") else str(interpreted.domain)
        sensor_val = interpreted.sensor.value if hasattr(interpreted.sensor, "value") else str(interpreted.sensor)
        task_seq_vals = [t.value if hasattr(t, "value") else str(t) for t in interpreted.task_sequence]
        return (
            f"=== SatQuery Auditable Execution Summary ===\n"
            f"Query: \"{interpreted.raw_query}\"\n"
            f"Domain: {domain_val} | Implied Sensor: {sensor_val}\n"
            f"Planned Task Sequence: {task_seq_vals}\n"
            f"Status: FAILED ({error_msg})"
        )

    def _synthesize_final_answer(self, task_outputs: List[TaskOutput]) -> str:
        """
        Combines individual task outputs into a unified response string.
        """
        if not task_outputs:
            return "No task outputs produced."
        if len(task_outputs) == 1:
            return str(task_outputs[0].raw_output)

        sections = []
        for out in task_outputs:
            sections.append(f"[{out.task_name.upper()}]: {out.raw_output}")
        return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Convenience Top-Level Function
# ---------------------------------------------------------------------------

def handle_query(
    raw_query: str,
    images: Optional[List[Any]] = None,
    interpreter: Optional[QueryInterpreter] = None,
    model_manager: Optional[ModelLifecycleManager] = None,
    inference_fn: Optional[Callable[[str, Any, RoutingDecision], Any]] = None,
) -> QueryResult:
    """
    Convenience top-level entrypoint for executing a SatQuery AI query end-to-end.
    """
    controller = QueryController(
        interpreter=interpreter,
        model_manager=model_manager,
        inference_fn=inference_fn,
    )
    return controller.handle_query(
        raw_query=raw_query,
        images=images,
    )
