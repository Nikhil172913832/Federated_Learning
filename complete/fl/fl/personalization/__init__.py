"""Personalized FL capabilities for heterogeneous clients with knowledge distillation."""

from .personalization import (
    PersonalizationManager,
    PersonalizationConfig,
    FineTuningStrategy,
)

# Knowledge Distillation Components
from .distillation import (
    DistillationConfig,
    DistillationLoss,
    LocalDistillationTrainer,
    ServerDistillationAggregator,
    ProgressiveDistillation,
    compute_distillation_metrics,
)

# Adapter Components
from .adapters import (
    LoRALayer,
    LoRALinear,
    BottleneckAdapter,
    AdapterLayer,
    PrefixTuningAdapter,
    ModelWithAdapters,
    count_parameters,
    print_adapter_summary,
)

# Personalized FL Components
from .personalized_client import (
    PersonalizedFLClient,
    PersonalizedClientConfig,
)

from .personalized_server import (
    PersonalizedFLServer,
    PersonalizedServerConfig,
)

from .orchestrator import (
    PersonalizedFLOrchestrator,
    PersonalizedFLConfig,
    create_personalized_fl_config,
)

__all__ = [
    # Legacy personalization
    "PersonalizationManager",
    "PersonalizationConfig",
    "FineTuningStrategy",
    # Distillation
    "DistillationConfig",
    "DistillationLoss",
    "LocalDistillationTrainer",
    "ServerDistillationAggregator",
    "ProgressiveDistillation",
    "compute_distillation_metrics",
    # Adapters
    "LoRALayer",
    "LoRALinear",
    "BottleneckAdapter",
    "AdapterLayer",
    "PrefixTuningAdapter",
    "ModelWithAdapters",
    "count_parameters",
    "print_adapter_summary",
    # Personalized FL
    "PersonalizedFLClient",
    "PersonalizedClientConfig",
    "PersonalizedFLServer",
    "PersonalizedServerConfig",
    "PersonalizedFLOrchestrator",
    "PersonalizedFLConfig",
    "create_personalized_fl_config",
]
