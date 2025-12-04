"""
Personalized Federated Learning Server with Knowledge Distillation.

Implements server-side orchestration for personalized FL:
- Client selection and scheduling
- Aggregation of shared parameters (FedAvg/FedProx)
- Model versioning and checkpointing
- Secure aggregation support
- Global model evaluation
- Anomaly detection for Byzantine clients
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import logging
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import json
import time

from fl.personalization.distillation import ServerDistillationAggregator, DistillationConfig

logger = logging.getLogger(__name__)


@dataclass
class PersonalizedServerConfig:
    """Configuration for personalized FL server."""
    # FL settings
    total_rounds: int = 1000
    clients_per_round: int = 50
    min_clients_per_round: int = 10
    
    # Aggregation strategy
    aggregation_strategy: str = "fedavg"  # "fedavg", "fedprox", "fedadam"
    weighted_aggregation: bool = True  # Weight by dataset size
    
    # Server-side optimization
    server_learning_rate: float = 1.0  # For FedAdam/FedYogi
    server_momentum: float = 0.9
    server_adam_beta1: float = 0.9
    server_adam_beta2: float = 0.999
    server_adam_epsilon: float = 1e-8
    
    # Robustness and security
    enable_anomaly_detection: bool = True
    anomaly_threshold_multiplier: float = 3.0  # Reject updates > median * X
    enable_secure_aggregation: bool = False
    enable_differential_privacy: bool = False
    dp_noise_multiplier: float = 0.1
    dp_clip_norm: float = 1.0
    
    # Model versioning and checkpointing
    checkpoint_every_n_rounds: int = 10
    keep_n_checkpoints: int = 5
    model_registry_path: Optional[Path] = None
    
    # Evaluation
    eval_every_n_rounds: int = 5
    eval_on_global_testset: bool = True
    
    # Client management
    client_timeout_seconds: float = 300.0
    max_client_failures: int = 3
    
    # Distillation (for server-side distillation)
    server_distillation_config: Optional[DistillationConfig] = None
    use_server_distillation: bool = False


class PersonalizedFLServer:
    """
    Personalized Federated Learning Server.
    
    Orchestrates FL training with personalization support:
    - Manages global model and rounds
    - Selects clients and coordinates training
    - Aggregates updates (shared parameters only)
    - Handles anomalies and security
    - Tracks metrics and manages checkpoints
    """
    
    def __init__(
        self,
        config: PersonalizedServerConfig,
        global_model: nn.Module,
        device: torch.device
    ):
        self.config = config
        self.device = device
        self.global_model = global_model.to(device)
        
        # Round tracking
        self.current_round = 0
        self.global_model_version = 0
        
        # Client management
        self.registered_clients = {}  # client_id -> client_info
        self.client_participation_history = defaultdict(list)
        self.client_performance_history = defaultdict(list)
        
        # Server-side optimizer state (for FedAdam, etc.)
        self.server_optimizer_state = self._initialize_server_optimizer()
        
        # Anomaly detection
        self.update_norm_history = []
        
        # Metrics tracking
        self.global_metrics_history = []
        
        # Checkpointing
        if config.model_registry_path:
            self.checkpoint_dir = Path(config.model_registry_path)
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.checkpoint_dir = None
            
        # Server-side distillation
        if config.use_server_distillation and config.server_distillation_config:
            self.distillation_aggregator = ServerDistillationAggregator(
                self.global_model,
                config.server_distillation_config,
                device
            )
        else:
            self.distillation_aggregator = None
            
        logger.info(
            f"Initialized PersonalizedFLServer with {config.total_rounds} rounds, "
            f"{config.clients_per_round} clients/round, "
            f"strategy={config.aggregation_strategy}"
        )
        
    def _initialize_server_optimizer(self) -> Dict:
        """Initialize server-side optimizer state (for FedAdam, etc.)."""
        if self.config.aggregation_strategy not in ["fedadam", "fedyogi"]:
            return {}
            
        state = {
            "momentum": {},  # First moment (m)
            "velocity": {},  # Second moment (v)
            "step": 0
        }
        
        for name, param in self.global_model.named_parameters():
            state["momentum"][name] = torch.zeros_like(param)
            state["velocity"][name] = torch.zeros_like(param)
            
        return state
        
    def register_client(
        self,
        client_id: str,
        client_info: Dict[str, Any]
    ):
        """
        Register a client with the server.
        
        Args:
            client_id: Unique client identifier
            client_info: Client metadata (dataset_size, capabilities, etc.)
        """
        self.registered_clients[client_id] = {
            "client_id": client_id,
            "registered_at": time.time(),
            "num_rounds_participated": 0,
            "total_samples": client_info.get("dataset_size", 0),
            "last_seen": time.time(),
            **client_info
        }
        logger.info(f"Registered client {client_id} with {client_info.get('dataset_size', 0)} samples")
        
    def select_clients(
        self,
        round_idx: int,
        available_clients: Optional[List[str]] = None
    ) -> List[str]:
        """
        Select clients for the current round.
        
        Args:
            round_idx: Current round index
            available_clients: List of available client IDs (None = all registered)
            
        Returns:
            List of selected client IDs
        """
        if available_clients is None:
            available_clients = list(self.registered_clients.keys())
            
        if len(available_clients) <= self.config.clients_per_round:
            selected = available_clients
        else:
            # Sampling strategy: uniform random (could be importance sampling)
            selected = np.random.choice(
                available_clients,
                size=self.config.clients_per_round,
                replace=False
            ).tolist()
            
        # Update participation tracking
        for client_id in selected:
            if client_id in self.registered_clients:
                self.registered_clients[client_id]["num_rounds_participated"] += 1
                self.registered_clients[client_id]["last_seen"] = time.time()
            self.client_participation_history[client_id].append(round_idx)
            
        logger.info(f"Round {round_idx}: Selected {len(selected)} clients")
        return selected
        
    def get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights (CPU tensors)."""
        return {
            name: param.detach().cpu().clone()
            for name, param in self.global_model.named_parameters()
        }
        
    def aggregate_updates(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Optional[Dict[str, float]] = None,
        round_idx: int = 0
    ) -> Dict[str, Any]:
        """
        Aggregate client updates into global model.
        
        Args:
            client_updates: Dict mapping client_id -> model_update
            client_weights: Optional weights for each client (e.g., dataset size)
            round_idx: Current round index
            
        Returns:
            Aggregation metrics and statistics
        """
        if not client_updates:
            logger.warning("No client updates to aggregate")
            return {"num_clients": 0}
            
        # Filter anomalous updates if enabled
        if self.config.enable_anomaly_detection:
            client_updates = self._filter_anomalous_updates(client_updates)
            
        if not client_updates:
            logger.error("All updates filtered as anomalous!")
            return {"num_clients": 0, "error": "all_anomalous"}
            
        # Compute aggregation weights
        if client_weights is None and self.config.weighted_aggregation:
            client_weights = self._compute_client_weights(client_updates)
        elif client_weights is None:
            # Uniform weighting
            n = len(client_updates)
            client_weights = {cid: 1.0 / n for cid in client_updates.keys()}
            
        # Normalize weights
        total_weight = sum(client_weights.values())
        client_weights = {k: v / total_weight for k, v in client_weights.items()}
        
        # Aggregate based on strategy
        if self.config.aggregation_strategy == "fedavg":
            aggregated = self._aggregate_fedavg(client_updates, client_weights)
        elif self.config.aggregation_strategy == "fedadam":
            aggregated = self._aggregate_fedadam(client_updates, client_weights, round_idx)
        elif self.config.aggregation_strategy == "fedprox":
            # FedProx uses FedAvg aggregation (proximal term is client-side)
            aggregated = self._aggregate_fedavg(client_updates, client_weights)
        else:
            raise ValueError(f"Unknown aggregation strategy: {self.config.aggregation_strategy}")
            
        # Apply differential privacy noise if enabled
        if self.config.enable_differential_privacy:
            aggregated = self._add_dp_noise(aggregated)
            
        # Update global model
        self._update_global_model(aggregated)
        
        # Compute aggregation statistics
        stats = self._compute_aggregation_stats(client_updates, client_weights)
        stats["num_clients"] = len(client_updates)
        stats["aggregation_strategy"] = self.config.aggregation_strategy
        
        self.current_round = round_idx
        self.global_model_version += 1
        
        return stats
        
    def _filter_anomalous_updates(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Filter out anomalous client updates."""
        # Compute L2 norm of each update
        update_norms = {}
        for client_id, update in client_updates.items():
            norm = torch.sqrt(
                sum(torch.sum(param ** 2) for param in update.values())
            ).item()
            update_norms[client_id] = norm
            
        if not update_norms:
            return client_updates
            
        # Compute median and threshold
        norms = list(update_norms.values())
        median_norm = np.median(norms)
        threshold = median_norm * self.config.anomaly_threshold_multiplier
        
        # Filter anomalies
        filtered = {}
        rejected = []
        
        for client_id, update in client_updates.items():
            if update_norms[client_id] <= threshold:
                filtered[client_id] = update
            else:
                rejected.append(client_id)
                logger.warning(
                    f"Rejected anomalous update from {client_id}: "
                    f"norm={update_norms[client_id]:.4f} > threshold={threshold:.4f}"
                )
                
        # Track update norms for adaptive thresholding
        self.update_norm_history.extend(norms)
        if len(self.update_norm_history) > 1000:
            self.update_norm_history = self.update_norm_history[-1000:]
            
        if rejected:
            logger.info(f"Filtered {len(rejected)} anomalous updates out of {len(client_updates)}")
            
        return filtered
        
    def _compute_client_weights(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, float]:
        """Compute client weights based on dataset size."""
        weights = {}
        
        for client_id in client_updates.keys():
            if client_id in self.registered_clients:
                weights[client_id] = float(self.registered_clients[client_id].get("total_samples", 1))
            else:
                weights[client_id] = 1.0
                
        return weights
        
    def _aggregate_fedavg(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Dict[str, float]
    ) -> Dict[str, torch.Tensor]:
        """FedAvg: Weighted average of client updates."""
        aggregated = {}
        
        # Get all parameter names
        param_names = list(next(iter(client_updates.values())).keys())
        
        for name in param_names:
            # Weighted sum
            weighted_sum = torch.zeros_like(
                client_updates[list(client_updates.keys())[0]][name]
            )
            
            for client_id, update in client_updates.items():
                if name in update:
                    weighted_sum += client_weights[client_id] * update[name]
                    
            aggregated[name] = weighted_sum
            
        return aggregated
        
    def _aggregate_fedadam(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Dict[str, float],
        round_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        FedAdam: Adaptive server-side optimization.
        
        Applies Adam optimizer on server side to aggregated pseudo-gradients.
        """
        # First get FedAvg aggregation as pseudo-gradient
        pseudo_grad = self._aggregate_fedavg(client_updates, client_weights)
        
        # Apply Adam update
        aggregated = {}
        beta1 = self.config.server_adam_beta1
        beta2 = self.config.server_adam_beta2
        eps = self.config.server_adam_epsilon
        lr = self.config.server_learning_rate
        
        self.server_optimizer_state["step"] += 1
        t = self.server_optimizer_state["step"]
        
        for name, grad in pseudo_grad.items():
            # Get momentum and velocity
            m = self.server_optimizer_state["momentum"][name]
            v = self.server_optimizer_state["velocity"][name]
            
            # Update biased first moment
            m = beta1 * m + (1 - beta1) * grad
            
            # Update biased second moment
            v = beta2 * v + (1 - beta2) * (grad ** 2)
            
            # Bias correction
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Compute update
            update = lr * m_hat / (torch.sqrt(v_hat) + eps)
            
            # Store updated moments
            self.server_optimizer_state["momentum"][name] = m
            self.server_optimizer_state["velocity"][name] = v
            
            aggregated[name] = update
            
        return aggregated
        
    def _add_dp_noise(
        self,
        aggregated: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to aggregated update."""
        noisy = {}
        
        for name, param in aggregated.items():
            # Gaussian noise proportional to clip norm
            noise = torch.randn_like(param) * (
                self.config.dp_clip_norm * self.config.dp_noise_multiplier
            )
            noisy[name] = param + noise
            
        logger.debug("Added DP noise to aggregated update")
        return noisy
        
    def _update_global_model(self, aggregated: Dict[str, torch.Tensor]):
        """Update global model with aggregated update."""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated:
                    param.copy_(aggregated[name].to(self.device))
                    
    def _compute_aggregation_stats(
        self,
        client_updates: Dict[str, Dict[str, torch.Tensor]],
        client_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Compute statistics about aggregation."""
        stats = {}
        
        # Update norm statistics
        update_norms = []
        for update in client_updates.values():
            norm = torch.sqrt(
                sum(torch.sum(param ** 2) for param in update.values())
            ).item()
            update_norms.append(norm)
            
        stats["mean_update_norm"] = np.mean(update_norms)
        stats["median_update_norm"] = np.median(update_norms)
        stats["std_update_norm"] = np.std(update_norms)
        stats["max_update_norm"] = np.max(update_norms)
        stats["min_update_norm"] = np.min(update_norms)
        
        # Weight statistics
        weights = list(client_weights.values())
        stats["mean_client_weight"] = np.mean(weights)
        stats["std_client_weight"] = np.std(weights)
        
        # Divergence statistics (cosine similarity between updates)
        if len(client_updates) > 1:
            similarities = []
            update_list = list(client_updates.values())
            
            for i in range(len(update_list)):
                for j in range(i + 1, len(update_list)):
                    sim = self._compute_update_similarity(update_list[i], update_list[j])
                    similarities.append(sim)
                    
            if similarities:
                stats["mean_update_similarity"] = np.mean(similarities)
                stats["std_update_similarity"] = np.std(similarities)
                
        return stats
        
    def _compute_update_similarity(
        self,
        update1: Dict[str, torch.Tensor],
        update2: Dict[str, torch.Tensor]
    ) -> float:
        """Compute cosine similarity between two updates."""
        # Flatten and concatenate all parameters
        vec1 = torch.cat([param.flatten() for param in update1.values()])
        vec2 = torch.cat([param.flatten() for param in update2.values()])
        
        # Cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            vec1.unsqueeze(0),
            vec2.unsqueeze(0)
        ).item()
        
        return similarity
        
    def evaluate_global_model(
        self,
        test_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate global model on test data."""
        self.global_model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.global_model(inputs)
                loss = nn.functional.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        metrics = {
            "global_loss": total_loss / len(test_loader),
            "global_accuracy": correct / total if total > 0 else 0.0
        }
        
        self.global_metrics_history.append({
            "round": self.current_round,
            **metrics
        })
        
        logger.info(
            f"Round {self.current_round} global evaluation: "
            f"loss={metrics['global_loss']:.4f}, acc={metrics['global_accuracy']:.4f}"
        )
        
        return metrics
        
    def save_checkpoint(self, round_idx: int):
        """Save model checkpoint."""
        if self.checkpoint_dir is None:
            return
            
        checkpoint_path = self.checkpoint_dir / f"round_{round_idx}.pt"
        
        checkpoint = {
            "round": round_idx,
            "model_version": self.global_model_version,
            "model_state_dict": self.global_model.state_dict(),
            "server_optimizer_state": self.server_optimizer_state,
            "config": self.config,
            "global_metrics_history": self.global_metrics_history,
            "timestamp": time.time()
        }
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Clean up old checkpoints
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only the most recent N."""
        if self.checkpoint_dir is None:
            return
            
        checkpoints = sorted(
            self.checkpoint_dir.glob("round_*.pt"),
            key=lambda p: int(p.stem.split("_")[1])
        )
        
        if len(checkpoints) > self.config.keep_n_checkpoints:
            for checkpoint in checkpoints[:-self.config.keep_n_checkpoints]:
                checkpoint.unlink()
                logger.debug(f"Removed old checkpoint {checkpoint}")
                
    def load_checkpoint(self, checkpoint_path: Path):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        self.current_round = checkpoint["round"]
        self.global_model_version = checkpoint["model_version"]
        self.server_optimizer_state = checkpoint.get("server_optimizer_state", {})
        self.global_metrics_history = checkpoint.get("global_metrics_history", [])
        
        logger.info(f"Loaded checkpoint from round {self.current_round}")
        
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training progress."""
        summary = {
            "current_round": self.current_round,
            "model_version": self.global_model_version,
            "num_registered_clients": len(self.registered_clients),
            "total_client_participations": sum(
                len(history) for history in self.client_participation_history.values()
            ),
            "global_metrics_history": self.global_metrics_history[-10:],  # Last 10 rounds
        }
        
        if self.global_metrics_history:
            latest = self.global_metrics_history[-1]
            summary["latest_global_accuracy"] = latest.get("global_accuracy", 0.0)
            summary["latest_global_loss"] = latest.get("global_loss", 0.0)
            
        return summary
        
    def export_metrics(self, output_path: Path):
        """Export training metrics to JSON."""
        metrics = {
            "config": {
                "total_rounds": self.config.total_rounds,
                "clients_per_round": self.config.clients_per_round,
                "aggregation_strategy": self.config.aggregation_strategy,
            },
            "global_metrics": self.global_metrics_history,
            "client_participation": {
                client_id: len(history)
                for client_id, history in self.client_participation_history.items()
            },
            "summary": self.get_training_summary()
        }
        
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
            
        logger.info(f"Exported metrics to {output_path}")
