"""
Knowledge distillation components for personalized federated learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import numpy as np
from dataclasses import dataclass


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    temperature: float = 2.0  # Temperature for softmax (T ∈ [1, 5])
    lambda_kd: float = 0.5  # Weight for KD loss (λ ∈ [0.2, 0.7])
    lambda_ce: float = 0.5  # Weight for CE loss (1 - λ)
    kd_loss_type: str = "kl"  # "kl", "mse", or "cosine"
    use_hard_labels: bool = True  # Use hard labels (CE) in addition to soft
    feature_distillation: bool = False  # Use feature-level distillation
    feature_lambda: float = 0.1  # Weight for feature distillation
    min_temperature: float = 1.0  # Minimum temperature
    max_temperature: float = 5.0  # Maximum temperature
    adaptive_temperature: bool = False  # Adapt temperature based on confidence


class DistillationLoss(nn.Module):
    """
    Combined distillation loss.
    
    Combines:
    1. Cross-entropy with hard labels (supervised loss)
    2. KL divergence with teacher soft targets (distillation loss)
    3. Optional feature-level distillation
    4. Optional regularization
    
    Loss formula:
    L = (1-λ) * L_CE + λ * T² * L_KD + μ * L_reg + β * L_feat
    """
    
    def __init__(self, config: DistillationConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.lambda_kd = config.lambda_kd
        self.lambda_ce = config.lambda_ce
        
        # Ensure lambdas sum to 1 for main losses
        total = self.lambda_kd + self.lambda_ce
        if total > 0:
            self.lambda_kd = self.lambda_kd / total
            self.lambda_ce = self.lambda_ce / total
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor,
        student_features: Optional[torch.Tensor] = None,
        teacher_features: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined distillation loss.
        
        Args:
            student_logits: Student model logits [batch, num_classes]
            teacher_logits: Teacher model logits [batch, num_classes]
            labels: Ground truth labels [batch]
            student_features: Optional student features for feature distillation
            teacher_features: Optional teacher features for feature distillation
            
        Returns:
            Tuple of (total_loss, loss_dict with breakdown)
        """
        losses = {}
        
        # 1. Cross-entropy loss with hard labels
        if self.config.use_hard_labels and self.lambda_ce > 0:
            ce_loss = F.cross_entropy(student_logits, labels)
            losses["ce"] = ce_loss.item()
        else:
            ce_loss = 0.0
            losses["ce"] = 0.0
            
        # 2. Knowledge distillation loss
        if self.lambda_kd > 0:
            # Adaptive temperature based on teacher confidence
            if self.config.adaptive_temperature:
                teacher_probs = F.softmax(teacher_logits, dim=-1)
                teacher_conf = teacher_probs.max(dim=-1)[0].mean()
                # Lower temperature for high confidence, higher for low confidence
                temp = self.temperature * (2.0 - teacher_conf.item())
                temp = np.clip(temp, self.config.min_temperature, self.config.max_temperature)
            else:
                temp = self.temperature
                
            kd_loss = self._compute_kd_loss(
                student_logits,
                teacher_logits,
                temperature=temp
            )
            losses["kd"] = kd_loss.item()
            losses["temperature"] = temp
        else:
            kd_loss = 0.0
            losses["kd"] = 0.0
            losses["temperature"] = self.temperature
            
        # 3. Feature distillation (optional)
        if (self.config.feature_distillation and 
            student_features is not None and 
            teacher_features is not None):
            feat_loss = self._compute_feature_loss(student_features, teacher_features)
            losses["feature"] = feat_loss.item()
        else:
            feat_loss = 0.0
            losses["feature"] = 0.0
            
        # Combine losses
        total_loss = (
            self.lambda_ce * ce_loss +
            self.lambda_kd * (temp ** 2) * kd_loss +  # T² scaling for gradient consistency
            self.config.feature_lambda * feat_loss
        )
        
        losses["total"] = total_loss.item()
        
        return total_loss, losses
        
    def _compute_kd_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute KD loss based on configured type."""
        if self.config.kd_loss_type == "kl":
            # KL divergence between soft targets
            student_soft = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
            kd_loss = F.kl_div(
                student_soft,
                teacher_soft,
                reduction="batchmean"
            )
        elif self.config.kd_loss_type == "mse":
            # MSE between logits
            kd_loss = F.mse_loss(
                student_logits / temperature,
                teacher_logits / temperature
            )
        elif self.config.kd_loss_type == "cosine":
            # Cosine similarity loss
            student_norm = F.normalize(student_logits, dim=-1)
            teacher_norm = F.normalize(teacher_logits, dim=-1)
            kd_loss = 1.0 - (student_norm * teacher_norm).sum(dim=-1).mean()
        else:
            raise ValueError(f"Unknown KD loss type: {self.config.kd_loss_type}")
            
        return kd_loss
        
    def _compute_feature_loss(
        self,
        student_features: torch.Tensor,
        teacher_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute feature-level distillation loss.
        
        Uses MSE between normalized features.
        """
        # Ensure same shape (may need projection if dimensions differ)
        if student_features.shape != teacher_features.shape:
            # Simple average pooling if spatial dimensions differ
            if len(student_features.shape) == 4:  # Conv features [B, C, H, W]
                student_features = F.adaptive_avg_pool2d(student_features, 1).flatten(1)
                teacher_features = F.adaptive_avg_pool2d(teacher_features, 1).flatten(1)
            
        # Normalize features
        student_norm = F.normalize(student_features, dim=-1)
        teacher_norm = F.normalize(teacher_features, dim=-1)
        
        # MSE loss
        return F.mse_loss(student_norm, teacher_norm)


class LocalDistillationTrainer:
    """
    Local distillation trainer for client-side training.
    
    Trains a student model locally using knowledge distillation from a global teacher.
    """
    
    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        distillation_config: DistillationConfig,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ):
        self.student = student_model
        self.teacher = teacher_model
        self.distillation_loss = DistillationLoss(distillation_config)
        self.optimizer = optimizer
        self.device = device
        
        # Freeze teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def train_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Single training step with distillation.
        
        Args:
            batch: (inputs, labels)
            
        Returns:
            Dictionary of loss values
        """
        self.student.train()
        
        inputs, labels = batch
        inputs = inputs.to(self.device)
        labels = labels.to(self.device)
        
        # Forward through teacher (no grad)
        with torch.no_grad():
            teacher_logits = self.teacher(inputs)
            
        # Forward through student
        student_logits = self.student(inputs)
        
        # Compute distillation loss
        loss, loss_dict = self.distillation_loss(
            student_logits,
            teacher_logits,
            labels
        )
        
        # Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (important for FL stability)
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        return loss_dict
        
    def train_epoch(
        self,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = {
            "total": 0.0,
            "ce": 0.0,
            "kd": 0.0,
            "feature": 0.0
        }
        
        num_batches = 0
        for batch in dataloader:
            loss_dict = self.train_step(batch)
            for key in epoch_losses:
                epoch_losses[key] += loss_dict.get(key, 0.0)
            num_batches += 1
            
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
            
        return epoch_losses


class ServerDistillationAggregator:
    """
    Server-side distillation aggregator (FedDF-style).
    
    Clients send logits on a public/auxiliary dataset.
    Server aggregates logits and distills into a global model.
    """
    
    def __init__(
        self,
        global_model: nn.Module,
        distillation_config: DistillationConfig,
        device: torch.device
    ):
        self.global_model = global_model
        self.config = distillation_config
        self.device = device
        
    def aggregate_logits(
        self,
        client_logits: List[torch.Tensor],
        client_weights: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Aggregate logits from multiple clients.
        
        Args:
            client_logits: List of logit tensors from clients [num_clients, batch, num_classes]
            client_weights: Optional weights for weighted averaging
            
        Returns:
            Aggregated logits [batch, num_classes]
        """
        if client_weights is None:
            client_weights = [1.0 / len(client_logits)] * len(client_logits)
            
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Weighted average of logits
        aggregated = torch.zeros_like(client_logits[0])
        for logits, weight in zip(client_logits, client_weights):
            aggregated += weight * logits
            
        return aggregated
        
    def distill_from_aggregated_logits(
        self,
        public_dataloader: torch.utils.data.DataLoader,
        aggregated_logits: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        num_epochs: int = 1
    ) -> Dict[str, float]:
        """
        Distill global model from aggregated logits.
        
        Args:
            public_dataloader: DataLoader for public/auxiliary dataset
            aggregated_logits: Pre-aggregated logits [batch, num_classes]
            optimizer: Optimizer for global model
            num_epochs: Number of distillation epochs
            
        Returns:
            Training metrics
        """
        self.global_model.train()
        distillation_loss = DistillationLoss(self.config)
        
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            for batch_idx, (inputs, labels) in enumerate(public_dataloader):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                # Get batch of aggregated logits
                batch_teacher_logits = aggregated_logits[
                    batch_idx * len(inputs):(batch_idx + 1) * len(inputs)
                ].to(self.device)
                
                # Forward through global model
                student_logits = self.global_model(inputs)
                
                # Compute distillation loss
                loss, _ = distillation_loss(
                    student_logits,
                    batch_teacher_logits,
                    labels
                )
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.global_model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
        return {"distillation_loss": total_loss / num_batches}


class ProgressiveDistillation:
    """
    Progressive distillation with curriculum learning.
    
    Gradually increases distillation weight and decreases temperature over training.
    """
    
    def __init__(
        self,
        initial_config: DistillationConfig,
        total_rounds: int,
        warmup_rounds: int = 10
    ):
        self.initial_config = initial_config
        self.total_rounds = total_rounds
        self.warmup_rounds = warmup_rounds
        self.current_round = 0
        
    def get_config_for_round(self, round_idx: int) -> DistillationConfig:
        """Get distillation config for current round."""
        self.current_round = round_idx
        config = DistillationConfig(**self.initial_config.__dict__)
        
        # Warmup phase: start with more CE, less KD
        if round_idx < self.warmup_rounds:
            warmup_ratio = round_idx / self.warmup_rounds
            config.lambda_kd = self.initial_config.lambda_kd * warmup_ratio
            config.lambda_ce = 1.0 - config.lambda_kd
        else:
            # Progressive phase: gradually decrease temperature
            progress = (round_idx - self.warmup_rounds) / (self.total_rounds - self.warmup_rounds)
            config.temperature = self.initial_config.temperature * (1.0 - 0.3 * progress)
            config.temperature = max(config.temperature, 1.0)
            
        return config


def compute_teacher_student_agreement(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor
) -> float:
    """
    Compute agreement rate between teacher and student predictions.
    
    Args:
        student_logits: [batch, num_classes]
        teacher_logits: [batch, num_classes]
        
    Returns:
        Agreement rate (0-1)
    """
    student_preds = student_logits.argmax(dim=-1)
    teacher_preds = teacher_logits.argmax(dim=-1)
    agreement = (student_preds == teacher_preds).float().mean()
    return agreement.item()


def compute_distillation_metrics(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 2.0
) -> Dict[str, float]:
    """
    Compute comprehensive distillation metrics.
    
    Returns metrics including:
    - Teacher accuracy
    - Student accuracy
    - Agreement rate
    - KL divergence
    - Jensen-Shannon divergence
    """
    metrics = {}
    
    # Accuracies
    teacher_preds = teacher_logits.argmax(dim=-1)
    student_preds = student_logits.argmax(dim=-1)
    
    metrics["teacher_acc"] = (teacher_preds == labels).float().mean().item()
    metrics["student_acc"] = (student_preds == labels).float().mean().item()
    metrics["agreement"] = (teacher_preds == student_preds).float().mean().item()
    
    # KL divergence
    student_soft = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
    kl_div = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
    metrics["kl_divergence"] = kl_div.item()
    
    # Jensen-Shannon divergence (symmetric)
    student_prob = F.softmax(student_logits / temperature, dim=-1)
    m = 0.5 * (teacher_soft + student_prob)
    js_div = 0.5 * (
        F.kl_div(torch.log(m), teacher_soft, reduction="batchmean") +
        F.kl_div(torch.log(m), student_prob, reduction="batchmean")
    )
    metrics["js_divergence"] = js_div.item()
    
    return metrics
