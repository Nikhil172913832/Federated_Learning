"""Advanced experiment management and comparison.

This module provides tools for:
- Comparing multiple experiments
- Hyperparameter importance analysis
- Model registry integration
- Experiment search and filtering
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

import pandas as pd

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


logger = logging.getLogger(__name__)


class ExperimentManager:
    """Advanced experiment management with MLflow."""

    def __init__(self, tracking_uri: Optional[str] = None):
        """Initialize experiment manager.
        
        Args:
            tracking_uri: MLflow tracking URI (defaults to env var or file:./mlruns)
        """
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow not available. Install with: pip install mlflow")

        self.tracking_uri = tracking_uri or mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(self.tracking_uri)
        self.client = MlflowClient(self.tracking_uri)

    def compare_experiments(
        self,
        experiment_names: List[str],
        metric_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare multiple experiments side-by-side.
        
        Args:
            experiment_names: List of experiment names to compare
            metric_names: Optional list of specific metrics to include
            
        Returns:
            DataFrame with comparison of experiments
        """
        all_runs = []

        for exp_name in experiment_names:
            try:
                experiment = self.client.get_experiment_by_name(exp_name)
                if experiment is None:
                    logger.warning(f"Experiment not found: {exp_name}")
                    continue

                runs = self.client.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["start_time DESC"],
                )

                for run in runs:
                    run_data = {
                        "experiment": exp_name,
                        "run_id": run.info.run_id,
                        "run_name": run.data.tags.get("mlflow.runName", ""),
                        "start_time": run.info.start_time,
                        "status": run.info.status,
                    }

                    # Add parameters
                    for key, value in run.data.params.items():
                        run_data[f"param_{key}"] = value

                    # Add metrics
                    for key, value in run.data.metrics.items():
                        if metric_names is None or key in metric_names:
                            run_data[f"metric_{key}"] = value

                    all_runs.append(run_data)

            except Exception as e:
                logger.error(f"Error processing experiment {exp_name}: {e}")

        if not all_runs:
            logger.warning("No runs found for comparison")
            return pd.DataFrame()

        df = pd.DataFrame(all_runs)
        return df

    def get_best_run(
        self,
        experiment_name: str,
        metric: str,
        mode: str = "max",
    ) -> Optional[Any]:
        """Get best run from an experiment.
        
        Args:
            experiment_name: Name of experiment
            metric: Metric to optimize
            mode: "max" or "min"
            
        Returns:
            Best run or None if not found
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning(f"Experiment not found: {experiment_name}")
                return None

            order = "DESC" if mode == "max" else "ASC"
            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} {order}"],
                max_results=1,
            )

            if runs:
                return runs[0]
            return None

        except Exception as e:
            logger.error(f"Error getting best run: {e}")
            return None

    def register_model(
        self,
        run_id: str,
        model_name: str,
        stage: str = "None",
    ) -> Optional[Any]:
        """Register model in MLflow Model Registry.
        
        Args:
            run_id: Run ID containing the model
            model_name: Name for the registered model
            stage: Stage to transition to ("None", "Staging", "Production")
            
        Returns:
            ModelVersion or None if registration failed
        """
        try:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri, model_name)

            logger.info(
                f"Registered model '{model_name}' version {mv.version} from run {run_id}"
            )

            # Transition to stage if specified
            if stage != "None":
                self.client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage=stage,
                )
                logger.info(f"Transitioned model to stage: {stage}")

            return mv

        except Exception as e:
            logger.error(f"Error registering model: {e}")
            return None

    def search_experiments(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
    ) -> List[Any]:
        """Search for runs matching filter criteria.
        
        Args:
            filter_string: MLflow filter string (e.g., "metrics.accuracy > 0.9")
            max_results: Maximum number of results
            
        Returns:
            List of matching runs
        """
        try:
            # Get all experiments
            experiments = self.client.search_experiments()
            exp_ids = [exp.experiment_id for exp in experiments]

            # Search runs
            runs = self.client.search_runs(
                experiment_ids=exp_ids,
                filter_string=filter_string or "",
                max_results=max_results,
                order_by=["start_time DESC"],
            )

            return runs

        except Exception as e:
            logger.error(f"Error searching experiments: {e}")
            return []

    def get_run_metrics_history(
        self,
        run_id: str,
        metric_name: str,
    ) -> pd.DataFrame:
        """Get metric history for a run.
        
        Args:
            run_id: Run ID
            metric_name: Name of metric
            
        Returns:
            DataFrame with metric history
        """
        try:
            history = self.client.get_metric_history(run_id, metric_name)
            
            data = [
                {
                    "step": m.step,
                    "value": m.value,
                    "timestamp": m.timestamp,
                }
                for m in history
            ]
            
            return pd.DataFrame(data)

        except Exception as e:
            logger.error(f"Error getting metric history: {e}")
            return pd.DataFrame()

    def delete_runs(
        self,
        experiment_name: str,
        filter_string: Optional[str] = None,
        dry_run: bool = True,
    ) -> int:
        """Delete runs matching criteria.
        
        Args:
            experiment_name: Name of experiment
            filter_string: Optional filter for runs to delete
            dry_run: If True, only print what would be deleted
            
        Returns:
            Number of runs deleted (or would be deleted if dry_run)
        """
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment is None:
                logger.warning(f"Experiment not found: {experiment_name}")
                return 0

            runs = self.client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string or "",
            )

            if dry_run:
                logger.info(f"Would delete {len(runs)} runs (dry run)")
                for run in runs:
                    logger.info(f"  - {run.info.run_id}: {run.data.tags.get('mlflow.runName', '')}")
                return len(runs)

            for run in runs:
                self.client.delete_run(run.info.run_id)
                logger.info(f"Deleted run: {run.info.run_id}")

            return len(runs)

        except Exception as e:
            logger.error(f"Error deleting runs: {e}")
            return 0
