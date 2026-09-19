"""
SatQuery AI — Scripts Package.
Contains all consolidated specialist modules, query interpreter,
model lifecycle manager, model registry, router, controller, and orchestrator.
"""

from .schema import Domain, InterpretedQuery, Sensor, Source, Task
from .keyword_fallback import classify_query_fallback
from .interpreter import QueryInterpreter
from .prompt_template import build_prompt
from .model_lifecycle_manager import ModelLifecycleManager, ModelSpec, Residency
from .model_registry import build_registry, InterpreterLLMWrapper
from .orchestrator import run_query
from .qwen_specialist import ChangeVQASpecialist
from .geochat_specialist import GeoChatSpecialist
from .fusion_model import FusionSpecialist
