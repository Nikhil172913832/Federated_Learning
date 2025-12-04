"""
Personalized Federated Learning Orchestrator.

Main entry point for running personalized FL experiments with knowledge distillation.
Coordinates server and clients, manages rounds, and tracks metrics.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import logging
from dataclasses import dataclass
import time
from tqdm import tqdm

from fl.personalization.personalized_client import PersonalizedFLClient, PersonalizedClientConfig
from fl.personalization.personalized_server import PersonalizedFLServer, PersonalizedServerConfig
from fl.personalization.distillation import DistillationConfig, ProgressiveDistillation
from fl.utils.metrics import MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class PersonalizedFLConfig:
    """Complete configuration for personalized FL experiment."""
    # Server config
    server_config: PersonalizedServerConfig
    
    # Client config template (will be customized per client)
    client_config_template: PersonalizedClientConfig
    
    # Distillation config
    distillation_config: DistillationConfig
    
    # Progressive distillation
    use_progressive_distillation: bool = True
    
    # Experiment settings
    experiment_name: str = "personalized_fl"
    output_dir: Path = Path("./outputs")
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Debug and logging
    verbose: bool = True
    log_level: str = "INFO"


class PersonalizedFLOrchestrator:
    """
    Main orchestrator for personalized federated learning experiments.
    
    Manages the full FL training pipeline:
    1. Initialize server and clients
    2. Run FL rounds (client selection, training, aggregation)
    3. Evaluate and checkpoint
    4. Track metrics and export results
    """
    
    def __init__(
        self,
        config: PersonalizedFLConfig,
        model_fn: Callable[[], nn.Module],
        client_data_loaders: Dict[str, Dict[str, torch.utils.data.DataLoader]],
        global_test_loader: Optional[torch.utils.data.DataLoader] = None
    ):
        """
        Args:
            config: Complete FL configuration
            model_fn: Function that returns a fresh model instance
            client_data_loaders: Dict mapping client_id -> {"train": loader, "val": loader}
            global_test_loader: Optional global test set for evaluation
        """
        self.config = config
        self.model_fn = model_fn
        self.client_data_loaders = client_data_loaders
        self.global_test_loader = global_test_loader
        
        # Set random seed
        torch.manual_seed(config.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.random_seed)
            
        # Setup device
        self.device = torch.device(config.device)
        
        # Create output directory
        self.output_dir = config.output_dir / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize server
        global_model = model_fn().to(self.device)
        self.server = PersonalizedFLServer(
            config.server_config,
            global_model,
            self.device
        )
        
        # Initialize clients
        self.clients = self._initialize_clients()
        
        # Progressive distillation
        if config.use_progressive_distillation:
            self.progressive_distillation = ProgressiveDistillation(
                config.distillation_config,
                config.server_config.total_rounds,
                warmup_rounds=config.client_config_template.personalization_warmup_rounds
            )
        else:
            self.progressive_distillation = None
            
        # Metrics tracking
        self.metrics_tracker = MetricsTracker(str(self.output_dir / "metrics.json"))
        
        logger.info(
            f"Initialized PersonalizedFLOrchestrator: {len(self.clients)} clients, "
            f"{config.server_config.total_rounds} rounds"
        )
        
    def _setup_logging(self):
        """Setup logging configuration."""
        log_file = self.output_dir / "training.log"
        
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
    def _initialize_clients(self) -> Dict[str, PersonalizedFLClient]:
        """Initialize all clients."""
        clients = {}
        
        for client_id, data_loaders in self.client_data_loaders.items():
            # Create client-specific config
            client_config = PersonalizedClientConfig(
                **{**self.config.client_config_template.__dict__, "client_id": client_id}
            )
            
            # Set storage path
            if client_config.storage_path is None:
                client_config.storage_path = self.output_dir / "personalization" / client_id
                
            # Set distillation config
            client_config.distillation_config = self.config.distillation_config
            
            # Create client
            base_model = self.model_fn()
            client = PersonalizedFLClient(client_config, base_model, self.device)
            
            # Set dataset size
            train_loader = data_loaders.get("train")
            if train_loader:
                client.set_dataset_size(len(train_loader.dataset))
                
            clients[client_id] = client
            
            # Register with server
            self.server.register_client(client_id, {
                "dataset_size": client.get_dataset_size(),
                "adapter_type": client_config.adapter_type
            })
            
        return clients
        
    def run(self) -> Dict[str, Any]:
        """
        Run the complete FL training process.
        
        Returns:
            Training results and metrics
        """
        logger.info(f"Starting personalized FL training: {self.config.experiment_name}")
        start_time = time.time()
        
        try:
            for round_idx in tqdm(
                range(self.config.server_config.total_rounds),
                desc="FL Rounds",
                disable=not self.config.verbose
            ):
                round_metrics = self._run_round(round_idx)
                
                # Track metrics
                self.metrics_tracker.log_metrics(round_idx, round_metrics)
                
                # Evaluate global model
                if (round_idx + 1) % self.config.server_config.eval_every_n_rounds == 0:
                    if self.global_test_loader:
                        global_metrics = self.server.evaluate_global_model(self.global_test_loader)
                        self.metrics_tracker.log_metrics(
                            round_idx,
                            {"global_eval": global_metrics}
                        )
                        
                # Checkpoint
                if (round_idx + 1) % self.config.server_config.checkpoint_every_n_rounds == 0:
                    self.server.save_checkpoint(round_idx)
                    
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed with error: {e}", exc_info=True)
            raise
        finally:
            # Save final results
            training_time = time.time() - start_time
            results = self._finalize_training(training_time)
            
        logger.info(f"Training completed in {training_time:.2f}s")
        return results
        
    def _run_round(self, round_idx: int) -> Dict[str, Any]:
        """Run a single FL round."""
        round_start = time.time()
        
        # Select clients
        available_clients = list(self.clients.keys())
        selected_client_ids = self.server.select_clients(round_idx, available_clients)
        
        # Get current distillation config (progressive)
        if self.progressive_distillation:
            current_distill_config = self.progressive_distillation.get_config_for_round(round_idx)
        else:
            current_distill_config = self.config.distillation_config
            
        # Update client distillation configs
        for client_id in selected_client_ids:
            self.clients[client_id].config.distillation_config = current_distill_config
            
        # Get global model weights
        global_weights = self.server.get_global_model_weights()
        
        # Client training
        client_updates = {}
        client_weights = {}
        client_metrics = {}
        
        for client_id in tqdm(
            selected_client_ids,
            desc=f"Round {round_idx} - Client Training",
            leave=False,
            disable=not self.config.verbose
        ):
            client = self.clients[client_id]
            
            # Send global model
            client.receive_global_model(global_weights, round_idx)
            
            # Local training
            train_loader = self.client_data_loaders[client_id]["train"]
            val_loader = self.client_data_loaders[client_id].get("val")
            
            metrics = client.local_train(train_loader, val_loader)
            
            # Get update
            update = client.get_model_update()
            
            client_updates[client_id] = update
            client_weights[client_id] = float(client.get_dataset_size())
            client_metrics[client_id] = metrics
            
        # Server aggregation
        aggregation_stats = self.server.aggregate_updates(
            client_updates,
            client_weights,
            round_idx
        )
        
        # Compile round metrics
        round_time = time.time() - round_start
        
        round_metrics = {
            "round": round_idx,
            "num_clients": len(selected_client_ids),
            "round_time": round_time,
            "aggregation": aggregation_stats,
            "client_metrics": self._aggregate_client_metrics(client_metrics),
            "distillation": {
                "temperature": current_distill_config.temperature,
                "lambda_kd": current_distill_config.lambda_kd,
            }
        }
        
        if self.config.verbose and round_idx % 10 == 0:
            logger.info(
                f"Round {round_idx}: "
                f"avg_loss={round_metrics['client_metrics'].get('avg_loss', 0):.4f}, "
                f"avg_acc={round_metrics['client_metrics'].get('avg_val_accuracy', 0):.4f}"
            )
            
        return round_metrics
        
    def _aggregate_client_metrics(
        self,
        client_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Aggregate metrics across clients."""
        if not client_metrics:
            return {}
            
        aggregated = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in client_metrics.values():
            all_keys.update(metrics.keys())
            
        # Aggregate each metric
        for key in all_keys:
            values = [m[key] for m in client_metrics.values() if key in m]
            if values:
                aggregated[f"avg_{key}"] = sum(values) / len(values)
                aggregated[f"std_{key}"] = (
                    sum((v - aggregated[f"avg_{key}"]) ** 2 for v in values) / len(values)
                ) ** 0.5
                aggregated[f"min_{key}"] = min(values)
                aggregated[f"max_{key}"] = max(values)
                
        return aggregated
        
    def _finalize_training(self, training_time: float) -> Dict[str, Any]:
        """Finalize training and save results."""
        # Save final checkpoint
        self.server.save_checkpoint(self.config.server_config.total_rounds - 1)
        
        # Export metrics
        self.metrics_tracker.save()
        self.server.export_metrics(self.output_dir / "server_metrics.json")
        
        # Get final evaluation
        final_metrics = {}
        if self.global_test_loader:
            final_metrics["global_test"] = self.server.evaluate_global_model(
                self.global_test_loader
            )
            
        # Evaluate each client on their local test set
        client_test_metrics = {}
        for client_id, client in self.clients.items():
            if "test" in self.client_data_loaders[client_id]:
                test_loader = self.client_data_loaders[client_id]["test"]
                test_metrics = client.evaluate(test_loader)
                client_test_metrics[client_id] = test_metrics
                
        if client_test_metrics:
            final_metrics["client_test"] = self._aggregate_client_metrics(client_test_metrics)
            
        # Compute personalization gains
        personalization_gains = self._compute_personalization_gains(client_test_metrics)
        if personalization_gains:
            final_metrics["personalization_gains"] = personalization_gains
            
        # Compile results
        results = {
            "config": {
                "experiment_name": self.config.experiment_name,
                "total_rounds": self.config.server_config.total_rounds,
                "num_clients": len(self.clients),
                "aggregation_strategy": self.config.server_config.aggregation_strategy,
            },
            "training_time": training_time,
            "final_metrics": final_metrics,
            "server_summary": self.server.get_training_summary(),
        }
        
        # Save results
        import json
        with open(self.output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"Results saved to {self.output_dir}")
        return results
        
    def _compute_personalization_gains(
        self,
        client_test_metrics: Dict[str, Dict[str, float]]
    ) -> Dict[str, float]:
        """Compute personalization gains (vs global model)."""
        if not client_test_metrics or not self.global_test_loader:
            return {}
            
        # Get global model performance on each client's data
        global_model_metrics = {}
        for client_id, data_loaders in self.client_data_loaders.items():
            if "test" in data_loaders:
                test_loader = data_loaders["test"]
                metrics = self._evaluate_model_on_loader(
                    self.server.global_model,
                    test_loader
                )
                global_model_metrics[client_id] = metrics
                
        # Compute gains
        gains = {}
        accuracies_personal = []
        accuracies_global = []
        
        for client_id in client_test_metrics:
            if client_id in global_model_metrics:
                personal_acc = client_test_metrics[client_id].get("accuracy", 0.0)
                global_acc = global_model_metrics[client_id].get("accuracy", 0.0)
                
                gain = personal_acc - global_acc
                gains[client_id] = gain
                
                accuracies_personal.append(personal_acc)
                accuracies_global.append(global_acc)
                
        if gains:
            import numpy as np
            gains["mean_gain"] = np.mean(list(gains.values()))
            gains["std_gain"] = np.std(list(gains.values()))
            gains["median_gain"] = np.median(list(gains.values()))
            gains["max_gain"] = max(gains.values())
            gains["min_gain"] = min(gains.values())
            
            gains["mean_personal_acc"] = np.mean(accuracies_personal)
            gains["mean_global_acc"] = np.mean(accuracies_global)
            
        return gains
        
    def _evaluate_model_on_loader(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate a model on a data loader."""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in dataloader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                logits = model(inputs)
                loss = nn.functional.cross_entropy(logits, labels)
                
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        return {
            "loss": total_loss / len(dataloader) if len(dataloader) > 0 else 0.0,
            "accuracy": correct / total if total > 0 else 0.0
        }


def create_personalized_fl_config(
    total_rounds: int = 100,
    clients_per_round: int = 10,
    local_epochs: int = 1,
    learning_rate: float = 1e-3,
    adapter_type: str = "lora",
    adapter_rank: int = 4,
    temperature: float = 2.0,
    lambda_kd: float = 0.5,
    **kwargs
) -> PersonalizedFLConfig:
    """
    Convenience function to create a PersonalizedFLConfig with common settings.
    
    Args:
        total_rounds: Total FL rounds
        clients_per_round: Clients selected per round
        local_epochs: Local training epochs per round
        learning_rate: Client learning rate
        adapter_type: "lora", "bottleneck", or "none"
        adapter_rank: LoRA rank
        temperature: Distillation temperature
        lambda_kd: Distillation weight
        **kwargs: Additional config overrides
        
    Returns:
        Complete PersonalizedFLConfig
    """
    server_config = PersonalizedServerConfig(
        total_rounds=total_rounds,
        clients_per_round=clients_per_round,
        **kwargs.get("server_config", {})
    )
    
    client_config = PersonalizedClientConfig(
        client_id="template",
        local_epochs=local_epochs,
        learning_rate=learning_rate,
        adapter_type=adapter_type,
        adapter_rank=adapter_rank,
        **kwargs.get("client_config", {})
    )
    
    distillation_config = DistillationConfig(
        temperature=temperature,
        lambda_kd=lambda_kd,
        lambda_ce=1.0 - lambda_kd,
        **kwargs.get("distillation_config", {})
    )
    
    config = PersonalizedFLConfig(
        server_config=server_config,
        client_config_template=client_config,
        distillation_config=distillation_config,
        **kwargs.get("fl_config", {})
    )
    
    return config
