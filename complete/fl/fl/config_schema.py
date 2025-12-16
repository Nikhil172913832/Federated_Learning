"""Configuration schema validation using Pydantic.

This module provides type-safe configuration validation with:
- Automatic type checking
- Range validation
- Custom validators
- Helpful error messages
"""

from typing import Optional, Dict, Any, List
from enum import Enum

try:
    from pydantic import BaseModel, Field, field_validator, model_validator
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    # Fallback for when Pydantic is not available
    class BaseModel:
        pass
    def Field(*args, **kwargs):
        return None
    def field_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def model_validator(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class PartitioningType(str, Enum):
    """Data partitioning strategies."""
    IID = "iid"
    LABEL_SKEW = "label_skew"
    QUANTITY_SKEW = "quantity_skew"
    COVARIATE_SHIFT = "covariate_shift"


class PersonalizationMethod(str, Enum):
    """Personalization methods."""
    NONE = "none"
    FEDPROX = "fedprox"
    FEDBN = "fedbn"
    FEDPER = "fedper"
    FINETUNE = "finetune"


class StorageBackend(str, Enum):
    """Storage backend options."""
    FOLDER = "folder"
    SQLITE = "sqlite"


class NonIIDConfig(BaseModel):
    """Non-IID data configuration."""
    type: PartitioningType
    params: Dict[str, Any] = Field(default_factory=dict)

    @field_validator('params')
    @classmethod
    def validate_params(cls, v, info):
        """Validate params based on partitioning type."""
        partition_type = info.data.get('type')
        
        if partition_type == PartitioningType.LABEL_SKEW:
            if 'alpha' in v:
                alpha = v['alpha']
                if not (0 < alpha <= 10):
                    raise ValueError(f"alpha must be in (0, 10], got {alpha}")
        
        return v


class DataConfig(BaseModel):
    """Data configuration."""
    dataset: str = "albertvillanova/medmnist-v2"
    subset: str = "pneumoniamnist"
    batch_size: int = Field(default=32, gt=0, le=1024)
    iid: bool = True
    non_iid: Optional[NonIIDConfig] = None

    @field_validator('batch_size')
    @classmethod
    def validate_batch_size(cls, v):
        """Warn if batch size is unusual."""
        if v not in [16, 32, 64, 128, 256]:
            import logging
            logging.getLogger(__name__).warning(
                f"Unusual batch size: {v}. Common values are 16, 32, 64, 128, 256"
            )
        return v


class PreprocessConfig(BaseModel):
    """Preprocessing configuration."""
    resize: Optional[List[int]] = None
    normalize_mean: List[float] = Field(default=[0.5])
    normalize_std: List[float] = Field(default=[0.5])
    augmentation: Dict[str, Any] = Field(default_factory=lambda: {"enabled": False, "params": {}})

    @field_validator('resize')
    @classmethod
    def validate_resize(cls, v):
        """Validate resize dimensions."""
        if v is not None:
            if len(v) != 2:
                raise ValueError(f"resize must be [height, width], got {v}")
            if any(dim <= 0 for dim in v):
                raise ValueError(f"resize dimensions must be positive, got {v}")
        return v


class DPSGDConfig(BaseModel):
    """Differential privacy configuration."""
    enabled: bool = False
    noise_multiplier: float = Field(default=0.8, gt=0, le=10)
    max_grad_norm: float = Field(default=1.0, gt=0, le=100)
    target_epsilon: Optional[float] = Field(default=None, gt=0)
    target_delta: float = Field(default=1e-5, gt=0, lt=1)

    @model_validator(mode='after')
    def validate_privacy_budget(self):
        """Validate privacy budget parameters."""
        if self.enabled and self.target_epsilon is not None:
            if self.target_epsilon > 10:
                import logging
                logging.getLogger(__name__).warning(
                    f"High epsilon value ({self.target_epsilon}) provides weak privacy guarantees"
                )
        return self


class PrivacyConfig(BaseModel):
    """Privacy configuration."""
    dp_sgd: DPSGDConfig = Field(default_factory=DPSGDConfig)


class PersonalizationConfig(BaseModel):
    """Personalization configuration."""
    method: PersonalizationMethod = PersonalizationMethod.NONE
    fedprox_mu: float = Field(default=0.0, ge=0, le=1)

    @model_validator(mode='after')
    def validate_fedprox_mu(self):
        """Validate FedProx mu parameter."""
        if self.method == PersonalizationMethod.FEDPROX and self.fedprox_mu == 0:
            import logging
            logging.getLogger(__name__).warning(
                "FedProx method selected but fedprox_mu is 0. Set fedprox_mu > 0 for proximal term."
            )
        return self


class StorageConfig(BaseModel):
    """Storage configuration."""
    backend: StorageBackend = StorageBackend.FOLDER
    root_dir: str = "./client_stores"
    sqlite_dir: str = "./client_sqlite"


class TopologyConfig(BaseModel):
    """Federated topology configuration."""
    num_clients: int = Field(default=10, ge=1, le=1000)
    fraction: float = Field(default=0.5, gt=0, le=1.0)

    @field_validator('fraction')
    @classmethod
    def validate_fraction(cls, v, info):
        """Validate fraction based on num_clients."""
        num_clients = info.data.get('num_clients', 10)
        min_clients = max(1, int(v * num_clients))
        
        if min_clients < 2:
            import logging
            logging.getLogger(__name__).warning(
                f"Fraction {v} with {num_clients} clients results in <2 clients per round"
            )
        
        return v


class TrainConfig(BaseModel):
    """Training configuration."""
    lr: float = Field(default=0.01, gt=0, le=1.0)
    local_epochs: int = Field(default=1, ge=1, le=100)
    num_server_rounds: int = Field(default=3, ge=1, le=1000)

    @field_validator('lr')
    @classmethod
    def validate_lr(cls, v):
        """Warn if learning rate is unusual."""
        if v > 0.1:
            import logging
            logging.getLogger(__name__).warning(
                f"High learning rate: {v}. This may cause training instability."
            )
        elif v < 0.0001:
            import logging
            logging.getLogger(__name__).warning(
                f"Very low learning rate: {v}. Training may be slow."
            )
        return v


class FLConfig(BaseModel):
    """Complete federated learning configuration."""
    seed: int = Field(default=42, ge=0)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    preprocess: PreprocessConfig = Field(default_factory=PreprocessConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    personalization: PersonalizationConfig = Field(default_factory=PersonalizationConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)

    class Config:
        extra = "forbid"  # Reject unknown fields
        validate_assignment = True  # Validate on assignment


def validate_config(config_dict: Dict[str, Any]) -> FLConfig:
    """Validate configuration dictionary.
    
    Args:
        config_dict: Configuration dictionary from YAML/JSON
        
    Returns:
        Validated FLConfig instance
        
    Raises:
        ValidationError: If configuration is invalid
    """
    if not PYDANTIC_AVAILABLE:
        import logging
        logging.getLogger(__name__).warning(
            "Pydantic not available, skipping config validation"
        )
        # Return a dummy object that acts like a dict
        class DictLike:
            def __init__(self, d):
                self.__dict__.update(d)
        return DictLike(config_dict)
    
    return FLConfig(**config_dict)


def validate_config_file(config_path: str) -> FLConfig:
    """Validate configuration file.
    
    Args:
        config_path: Path to YAML/JSON config file
        
    Returns:
        Validated FLConfig instance
    """
    from fl.config import load_run_config
    
    config_dict = load_run_config(config_path)
    return validate_config(config_dict)
