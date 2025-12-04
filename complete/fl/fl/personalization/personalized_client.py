"""
Personalized Federated Learning Client with Knowledge Distillation.

Implements client-side training with:
- Knowledge distillation from global teacher
- Adapter-based personalization (LoRA/bottleneck)
- Local personalization storage
- Update clipping and compression
- Heterogeneity handling
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import numpy as np

from fl.personalization.adapters import (
    ModelWithAdapters,
    LoRALinear,
    BottleneckAdapter,
    count_parameters
)
from fl.personalization.distillation import (
    DistillationConfig,
    LocalDistillationTrainer,
    compute_distillation_metrics
)
from fl.personalization_storage import PersonalizationStorage

logger = logging.getLogger(__name__)


@dataclass
class PersonalizedClientConfig:
    """Configuration for personalized FL client."""
    # Client identification
    client_id: str
    
    # Training hyperparameters
    local_epochs: int = 1
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    
    # Adapter configuration
    adapter_type: str = "lora"  # "lora", "bottleneck", or "none"
    adapter_rank: int = 4  # For LoRA
    adapter_alpha: float = 16.0  # For LoRA
    adapter_bottleneck_dim: int = 64  # For bottleneck adapters
    adapter_dropout: float = 0.0
    target_modules: List[str] = None  # Modules to adapt (None = auto)
    
    # Distillation configuration
    distillation_config: DistillationConfig = None
    
    # Update clipping and compression
    clip_norm: float = 1.0  # L2 norm clipping threshold
    gradient_clipping: bool = True
    compress_updates: bool = False
    compression_ratio: float = 0.1  # Top-k sparsification ratio
    quantize_bits: int = 8  # Quantization bits (0 = no quantization)
    
    # Personalization settings
    min_samples_for_personalization: int = 50
    freeze_backbone: bool = True  # Freeze base model, train only adapters
    personalization_warmup_rounds: int = 5
    
    # Regularization
    use_regularization: bool = False
    reg_lambda: float = 1e-4  # Regularization weight
    
    # Storage
    storage_path: Optional[Path] = None


class PersonalizedFLClient:
    """
    Personalized Federated Learning Client.
    
    Handles local training with knowledge distillation and adapter-based personalization.
    """
    
    def __init__(
        self,
        config: PersonalizedClientConfig,
        base_model: nn.Module,
        device: torch.device
    ):
        self.config = config
        self.device = device
        self.round_idx = 0
        
        # Initialize base model with adapters
        if config.adapter_type != "none":
            adapter_config = {
                "type": config.adapter_type,
                "rank": config.adapter_rank,
                "alpha": config.adapter_alpha,
                "bottleneck_dim": config.adapter_bottleneck_dim,
                "dropout": config.adapter_dropout
            }
            self.model = ModelWithAdapters(
                base_model,
                adapter_config,
                target_modules=config.target_modules
            )
            if config.freeze_backbone:
                self.model.freeze_base_model()
        else:
            self.model = base_model
            
        self.model = self.model.to(device)
        
        # Initialize personalization storage
        if config.storage_path:
            self.storage = PersonalizationStorage(
                str(config.storage_path),
                config.client_id
            )
            self._load_personalized_params()
        else:
            self.storage = None
            
        # Training components
        self.optimizer = self._create_optimizer()
        self.teacher_model = None  # Set when receiving global model
        self.trainer = None
        
        # Metrics tracking
        self.metrics_history = []
        
        logger.info(
            f"Initialized PersonalizedFLClient {config.client_id} "
            f"with {count_parameters(self.model, trainable_only=True):,} trainable params"
        )
        
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for local training."""
        # Only optimize trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        return optimizer
        
    def receive_global_model(
        self,
        global_weights: Dict[str, torch.Tensor],
        round_idx: int
    ):
        """
        Receive global model weights from server.
        
        Updates shared parameters while keeping personalized parameters local.
        
        Args:
            global_weights: Dictionary of global model weights
            round_idx: Current FL round
        """
        self.round_idx = round_idx
        
        # Update shared parameters only
        model_dict = self.model.state_dict()
        
        for name, param in global_weights.items():
            # Only update non-adapter/non-personalized parameters
            if name in model_dict:
                if self.config.adapter_type != "none":
                    # Skip adapter parameters
                    if "lora" not in name and "adapter" not in name and "prefix" not in name:
                        model_dict[name] = param.to(self.device)
                else:
                    model_dict[name] = param.to(self.device)
                    
        self.model.load_state_dict(model_dict, strict=False)
        
        # Create teacher model (frozen copy of global model for distillation)
        if self.config.distillation_config is not None:
            self.teacher_model = self._create_teacher_model(global_weights)
            
        logger.info(f"Client {self.config.client_id} received global model for round {round_idx}")
        
    def _create_teacher_model(self, global_weights: Dict[str, torch.Tensor]) -> nn.Module:
        """Create frozen teacher model from global weights."""
        # Create a copy of base model architecture
        if isinstance(self.model, ModelWithAdapters):
            teacher = type(self.model.base_model)()
        else:
            teacher = type(self.model)()
            
        teacher.load_state_dict(global_weights, strict=False)
        teacher = teacher.to(self.device)
        teacher.eval()
        
        for param in teacher.parameters():
            param.requires_grad = False
            
        return teacher
        
    def local_train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader] = None
    ) -> Dict[str, Any]:
        """
        Perform local training with knowledge distillation.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            
        Returns:
            Training metrics and statistics
        """
        # Check if we have enough data for personalization
        dataset_size = len(train_loader.dataset)
        enable_personalization = (
            dataset_size >= self.config.min_samples_for_personalization and
            self.round_idx >= self.config.personalization_warmup_rounds
        )
        
        if not enable_personalization and self.config.adapter_type != "none":
            logger.warning(
                f"Client {self.config.client_id}: Insufficient data ({dataset_size}) "
                f"or warmup not complete. Personalization disabled for this round."
            )
            
        # Initialize trainer
        if self.config.distillation_config and self.teacher_model is not None:
            self.trainer = LocalDistillationTrainer(
                self.model,
                self.teacher_model,
                self.config.distillation_config,
                self.optimizer,
                self.device
            )
        
        # Training loop
        epoch_metrics = []
        for epoch in range(self.config.local_epochs):
            if self.trainer:
                metrics = self.trainer.train_epoch(train_loader)
            else:
                metrics = self._train_epoch_standard(train_loader)
                
            epoch_metrics.append(metrics)
            
        # Average metrics across epochs
        avg_metrics = {
            key: np.mean([m[key] for m in epoch_metrics])
            for key in epoch_metrics[0].keys()
        }
        
        # Validation
        if val_loader:
            val_metrics = self.evaluate(val_loader)
            avg_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
            
        # Compute update statistics
        update_stats = self._compute_update_statistics()
        avg_metrics.update(update_stats)
        
        # Save personalized parameters
        if enable_personalization and self.storage:
            self._save_personalized_params()
            
        self.metrics_history.append(avg_metrics)
        
        logger.info(
            f"Client {self.config.client_id} completed local training: "
            f"loss={avg_metrics.get('total', 0.0):.4f}, "
            f"acc={avg_metrics.get('val_accuracy', 0.0):.4f}"
        )
        
        return avg_metrics
        
    def _train_epoch_standard(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Standard training epoch without distillation."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            # Forward
            logits = self.model(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
            
            # Add regularization if enabled
            if self.config.use_regularization:
                reg_loss = self._compute_regularization_loss()
                loss = loss + self.config.reg_lambda * reg_loss
                
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.clip_norm
                )
                
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        return {"total": total_loss / num_batches}
        
    def _compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss to prevent catastrophic personalization.
        
        Penalizes deviation of shared parameters from global model.
        """
        if self.teacher_model is None:
            return torch.tensor(0.0, device=self.device)
            
        reg_loss = 0.0
        student_dict = self.model.state_dict()
        teacher_dict = self.teacher_model.state_dict()
        
        for name, student_param in student_dict.items():
            # Only regularize shared parameters
            if name in teacher_dict and "adapter" not in name and "lora" not in name:
                teacher_param = teacher_dict[name]
                reg_loss += torch.norm(student_param - teacher_param, p=2)
                
        return reg_loss
        
    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model on validation/test data."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(inputs)
                loss = nn.functional.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return {
            "loss": total_loss / len(dataloader),
            "accuracy": correct / total if total > 0 else 0.0
        }
        
    def get_model_update(self) -> Dict[str, torch.Tensor]:
        """
        Get model update to send to server.
        
        Returns only shared parameters (not personalized adapters).
        Applies clipping and compression if configured.
        """
        # Extract shared parameters only
        update = {}
        for name, param in self.model.named_parameters():
            # Skip adapter/personalized parameters
            if "lora" in name or "adapter" in name or "prefix" in name:
                continue
            update[name] = param.detach().cpu().clone()
            
        # Apply clipping to update
        if self.config.gradient_clipping:
            update = self._clip_update(update)
            
        # Apply compression
        if self.config.compress_updates:
            update = self._compress_update(update)
            
        return update
        
    def _clip_update(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Clip update to maximum L2 norm."""
        # Compute total L2 norm
        total_norm = torch.sqrt(
            sum(torch.sum(param ** 2) for param in update.values())
        )
        
        clip_coef = self.config.clip_norm / (total_norm + 1e-6)
        
        if clip_coef < 1.0:
            # Clip all parameters proportionally
            update = {
                name: param * clip_coef
                for name, param in update.items()
            }
            logger.debug(f"Clipped update: norm {total_norm:.4f} -> {self.config.clip_norm}")
            
        return update
        
    def _compress_update(self, update: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compress update using sparsification and/or quantization."""
        compressed = {}
        
        for name, param in update.items():
            # Top-k sparsification
            if self.config.compression_ratio < 1.0:
                param_flat = param.flatten()
                k = max(1, int(len(param_flat) * self.config.compression_ratio))
                topk_vals, topk_idx = torch.topk(param_flat.abs(), k)
                
                sparse_param = torch.zeros_like(param_flat)
                sparse_param[topk_idx] = param_flat[topk_idx]
                param = sparse_param.reshape(param.shape)
                
            # Quantization
            if self.config.quantize_bits > 0 and self.config.quantize_bits < 32:
                param = self._quantize_tensor(param, self.config.quantize_bits)
                
            compressed[name] = param
            
        return compressed
        
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """Quantize tensor to specified number of bits."""
        # Simple uniform quantization
        min_val = tensor.min()
        max_val = tensor.max()
        
        n_levels = 2 ** bits
        scale = (max_val - min_val) / n_levels
        
        quantized = torch.round((tensor - min_val) / scale)
        dequantized = quantized * scale + min_val
        
        return dequantized
        
    def _compute_update_statistics(self) -> Dict[str, float]:
        """Compute statistics about the model update."""
        stats = {}
        
        update = self.get_model_update()
        
        # L2 norm of update
        update_norm = torch.sqrt(
            sum(torch.sum(param ** 2) for param in update.values())
        ).item()
        stats["update_norm"] = update_norm
        
        # Number of parameters in update
        stats["update_params"] = sum(p.numel() for p in update.values())
        
        # Sparsity (if compressed)
        if self.config.compress_updates:
            total_elements = sum(p.numel() for p in update.values())
            nonzero_elements = sum((p != 0).sum().item() for p in update.values())
            stats["update_sparsity"] = 1.0 - (nonzero_elements / total_elements)
        else:
            stats["update_sparsity"] = 0.0
            
        return stats
        
    def _save_personalized_params(self):
        """Save personalized parameters to storage."""
        if self.storage is None:
            return
            
        # Get adapter parameters
        if isinstance(self.model, ModelWithAdapters):
            adapter_params = self.model.get_adapter_params()
        else:
            adapter_params = {}
            
        # Save to storage
        self.storage.save_params(
            round_id=self.round_idx,
            params=adapter_params,
            metadata={
                "dataset_size": self.config.min_samples_for_personalization,
                "adapter_type": self.config.adapter_type
            }
        )
        
    def _load_personalized_params(self):
        """Load personalized parameters from storage."""
        if self.storage is None:
            return
            
        try:
            params, metadata = self.storage.load_latest_params()
            
            if params and isinstance(self.model, ModelWithAdapters):
                self.model.load_adapter_params(params)
                logger.info(
                    f"Loaded personalized params from round {metadata.get('round_id', 'unknown')}"
                )
        except Exception as e:
            logger.warning(f"Failed to load personalized params: {e}")
            
    def get_dataset_size(self) -> int:
        """Get size of client's local dataset (for weighted aggregation)."""
        # This would be set during data loading
        return getattr(self, '_dataset_size', 0)
        
    def set_dataset_size(self, size: int):
        """Set size of client's local dataset."""
        self._dataset_size = size
