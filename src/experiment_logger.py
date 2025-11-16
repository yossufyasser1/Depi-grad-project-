import mlflow
import mlflow.pytorch
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
import torch

# Import project config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentLogger:
    """MLflow experiment logger for tracking model training and evaluation."""
    
    def __init__(self, experiment_name: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize MLflow experiment logger.
        
        Args:
            experiment_name: Name of the experiment (uses config default if not provided)
            config: Configuration object
        """
        self.config = config or Config()
        
        # Use config values
        if experiment_name is None:
            experiment_name = self.config.EXPERIMENT_NAME
        
        tracking_uri = self.config.MLFLOW_TRACKING_URI
        
        # Set up MLflow
        mlflow.set_tracking_uri(tracking_uri)
        
        # Create experiment if it doesn't exist
        try:
            self.experiment = mlflow.set_experiment(experiment_name)
            self.experiment_name = experiment_name
            logger.info(f'MLflow experiment set to: {experiment_name}')
            logger.info(f'Tracking URI: {tracking_uri}')
            logger.info(f'Experiment ID: {self.experiment.experiment_id}')
        except Exception as e:
            logger.error(f'Error setting up MLflow experiment: {e}')
            raise
        
        self.current_run = None
    
    def start_run(self, run_name: Optional[str] = None, tags: Optional[Dict[str, str]] = None):
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            tags: Dictionary of tags to add to the run
        """
        if self.current_run is not None:
            logger.warning('A run is already active. Ending it before starting a new one.')
            self.end_run()
        
        self.current_run = mlflow.start_run(run_name=run_name, tags=tags)
        logger.info(f'Started MLflow run: {self.current_run.info.run_id}')
        return self.current_run
    
    def end_run(self, status: str = 'FINISHED'):
        """
        End the current MLflow run.
        
        Args:
            status: Status of the run ('FINISHED', 'FAILED', 'KILLED')
        """
        if self.current_run is not None:
            mlflow.end_run(status=status)
            logger.info(f'Ended MLflow run with status: {status}')
            self.current_run = None
        else:
            logger.warning('No active run to end')
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log parameters to current run.
        
        Args:
            params: Dictionary of parameters to log
        """
        if self.current_run is None:
            logger.warning('No active run. Parameters not logged.')
            return
        
        # Flatten nested dictionaries
        flat_params = self._flatten_dict(params)
        
        # MLflow has a limit on parameter length, truncate if needed
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:497] + '...'
            mlflow.log_param(key, str_value)
        
        logger.info(f'Logged {len(flat_params)} parameters')
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log metrics to current run.
        
        Args:
            metrics: Dictionary of metrics to log
            step: Optional step number for the metrics
        """
        if self.current_run is None:
            logger.warning('No active run. Metrics not logged.')
            return
        
        logged_count = 0
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                try:
                    mlflow.log_metric(metric_name, float(metric_value), step=step)
                    logged_count += 1
                except Exception as e:
                    logger.warning(f'Could not log metric {metric_name}: {e}')
        
        logger.info(f'Logged {logged_count} metrics' + (f' at step {step}' if step else ''))
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """
        Log a single metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Optional step number
        """
        self.log_metrics({key: value}, step=step)
    
    def log_model(self, model: torch.nn.Module, artifact_path: str = 'model', **kwargs):
        """
        Log PyTorch model to MLflow.
        
        Args:
            model: PyTorch model to log
            artifact_path: Path within the run to store the model
            **kwargs: Additional arguments for mlflow.pytorch.log_model
        """
        if self.current_run is None:
            logger.warning('No active run. Model not logged.')
            return
        
        try:
            mlflow.pytorch.log_model(model, artifact_path, **kwargs)
            logger.info(f'Model logged to {artifact_path}')
        except Exception as e:
            logger.error(f'Error logging model: {e}')
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """
        Log artifacts (files/folders) to MLflow.
        
        Args:
            local_dir: Local directory containing artifacts
            artifact_path: Path within the run to store artifacts
        """
        if self.current_run is None:
            logger.warning('No active run. Artifacts not logged.')
            return
        
        try:
            mlflow.log_artifacts(local_dir, artifact_path)
            logger.info(f'Artifacts logged from {local_dir}')
        except Exception as e:
            logger.error(f'Error logging artifacts: {e}')
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log a single artifact file to MLflow.
        
        Args:
            local_path: Local path to artifact file
            artifact_path: Path within the run to store artifact
        """
        if self.current_run is None:
            logger.warning('No active run. Artifact not logged.')
            return
        
        try:
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f'Artifact logged: {local_path}')
        except Exception as e:
            logger.error(f'Error logging artifact: {e}')
    
    def log_training_run(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float],
        model: Optional[torch.nn.Module] = None,
        artifact_path: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Log a complete training run (params, metrics, model, artifacts).
        
        Args:
            params: Training parameters
            metrics: Final metrics
            model: Trained model
            artifact_path: Path to artifacts directory
            run_name: Name for the run
            tags: Tags for the run
        """
        with mlflow.start_run(run_name=run_name, tags=tags) as run:
            self.current_run = run
            
            # Log parameters
            self.log_params(params)
            
            # Log metrics
            self.log_metrics(metrics)
            
            # Log model if provided
            if model is not None:
                self.log_model(model)
            
            # Log artifacts if provided
            if artifact_path is not None and Path(artifact_path).exists():
                self.log_artifacts(artifact_path)
            
            logger.info(f'Training run logged: {run.info.run_id}')
            self.current_run = None
    
    def log_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Optional[Dict[str, float]] = None):
        """
        Log metrics for a training epoch.
        
        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Log training metrics with prefix
        train_metrics_prefixed = {f'train_{k}': v for k, v in train_metrics.items()}
        self.log_metrics(train_metrics_prefixed, step=epoch)
        
        # Log validation metrics with prefix
        if val_metrics:
            val_metrics_prefixed = {f'val_{k}': v for k, v in val_metrics.items()}
            self.log_metrics(val_metrics_prefixed, step=epoch)
    
    def compare_runs(self, metric_name: str, top_n: int = 5, ascending: bool = False) -> List[Dict]:
        """
        Compare top N runs by a specific metric.
        
        Args:
            metric_name: Name of the metric to compare
            top_n: Number of top runs to return
            ascending: Sort in ascending order (True) or descending (False)
            
        Returns:
            List of run information dictionaries
        """
        try:
            # Search runs in the experiment
            runs = mlflow.search_runs(
                experiment_ids=[self.experiment.experiment_id],
                order_by=[f'metrics.{metric_name} {"ASC" if ascending else "DESC"}']
            )
            
            # Get top N runs
            top_runs = runs.head(top_n)
            
            logger.info(f'Found {len(runs)} runs, returning top {len(top_runs)}')
            
            return top_runs.to_dict('records')
            
        except Exception as e:
            logger.error(f'Error comparing runs: {e}')
            return []
    
    def get_best_run(self, metric_name: str, ascending: bool = False) -> Optional[Dict]:
        """
        Get the best run based on a metric.
        
        Args:
            metric_name: Name of the metric
            ascending: Whether lower is better (True) or higher is better (False)
            
        Returns:
            Best run information dictionary
        """
        runs = self.compare_runs(metric_name, top_n=1, ascending=ascending)
        return runs[0] if runs else None
    
    def load_model(self, run_id: str, artifact_path: str = 'model') -> torch.nn.Module:
        """
        Load a logged model from a run.
        
        Args:
            run_id: MLflow run ID
            artifact_path: Path to model within the run
            
        Returns:
            Loaded PyTorch model
        """
        try:
            model_uri = f'runs:/{run_id}/{artifact_path}'
            model = mlflow.pytorch.load_model(model_uri)
            logger.info(f'Loaded model from run {run_id}')
            return model
        except Exception as e:
            logger.error(f'Error loading model: {e}')
            raise
    
    def get_run_info(self, run_id: str) -> Dict:
        """
        Get information about a specific run.
        
        Args:
            run_id: MLflow run ID
            
        Returns:
            Run information dictionary
        """
        try:
            run = mlflow.get_run(run_id)
            return {
                'run_id': run.info.run_id,
                'experiment_id': run.info.experiment_id,
                'status': run.info.status,
                'start_time': run.info.start_time,
                'end_time': run.info.end_time,
                'params': run.data.params,
                'metrics': run.data.metrics,
                'tags': run.data.tags
            }
        except Exception as e:
            logger.error(f'Error getting run info: {e}')
            return {}
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """
        Flatten nested dictionary for MLflow parameter logging.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for recursion
            sep: Separator for nested keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def set_tags(self, tags: Dict[str, str]):
        """
        Set tags for the current run.
        
        Args:
            tags: Dictionary of tags
        """
        if self.current_run is None:
            logger.warning('No active run. Tags not set.')
            return
        
        for key, value in tags.items():
            mlflow.set_tag(key, value)
        
        logger.info(f'Set {len(tags)} tags')

# Usage:
# # Initialize logger with config
# exp_logger = ExperimentLogger(experiment_name='my_experiment')
#
# # Log complete training run
# exp_logger.log_training_run(
#     params={'learning_rate': 0.001, 'batch_size': 32},
#     metrics={'accuracy': 0.95, 'loss': 0.05},
#     model=trained_model,
#     run_name='experiment_1'
# )
#
# # Or log incrementally
# exp_logger.start_run(run_name='training_run')
# exp_logger.log_params({'lr': 0.001})
# for epoch in range(10):
#     exp_logger.log_epoch(epoch, {'loss': 0.5}, {'val_loss': 0.6})
# exp_logger.log_model(model)
# exp_logger.end_run()
#
# # Compare runs
# best_runs = exp_logger.compare_runs('accuracy', top_n=5)
# best_run = exp_logger.get_best_run('accuracy', ascending=False)
