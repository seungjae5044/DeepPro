import wandb
import logging
import os
from typing import Dict, Any, Optional

class WandbLogger:
    def __init__(self, config: Dict[str, Any], project_name: str = "deep_pro_experiments"):
        self.config = self._sanitize_config(config)
        self.project_name = project_name
        self.run = None
        
        # Setup standard logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        if config.get('logging', {}).get('use_wandb', False):
            self.init_wandb()
    
    def _sanitize_config(self, config) -> Dict[str, Any]:
        """Sanitize config to make it JSON serializable for WandB"""
        import json
        from omegaconf import DictConfig, OmegaConf
        
        def convert_value(obj):
            if isinstance(obj, DictConfig):
                return {str(k): convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {str(k): convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_value(item) for item in obj]
            elif isinstance(obj, (str, int, float, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        try:
            if isinstance(config, DictConfig):
                config = OmegaConf.to_container(config, resolve=True)
            
            sanitized = convert_value(config)
            # Test if it's JSON serializable
            json.dumps(sanitized)
            return sanitized
        except (TypeError, ValueError, AttributeError):
            # If still not serializable, return a simplified version
            try:
                return {
                    "experiment_name": config.get('experiment', {}).get('name', 'unknown') if hasattr(config, 'get') else 'unknown',
                    "model_type": str(config.get('model', {}).get('_target_', 'unknown')) if hasattr(config, 'get') else 'unknown',
                    "optimizer_type": str(config.get('optimizer', {}).get('_target_', 'unknown')) if hasattr(config, 'get') else 'unknown',
                    "scheduler_type": str(config.get('scheduler', {}).get('_target_', 'unknown')) if hasattr(config, 'get') else 'unknown'
                }
            except:
                return {"config": "failed_to_serialize"}
    
    def init_wandb(self):
        """Initialize Weights & Biases logging"""
        try:
            # Extract safe values for tags
            model_name = "unknown"
            if self.config.get('model', {}).get('_target_'):
                model_name = str(self.config['model']['_target_']).split('.')[-1]
            
            optimizer_name = "unknown"
            if self.config.get('optimizer', {}).get('_target_'):
                optimizer_name = str(self.config['optimizer']['_target_']).split('.')[-1]
                
            scheduler_name = "unknown"
            if self.config.get('scheduler', {}).get('_target_'):
                scheduler_name = str(self.config['scheduler']['_target_']).split('.')[-1]
            
            experiment_name = self.config.get('experiment', {}).get('name', 'experiment')
            
            self.run = wandb.init(
                project=f"{self.project_name}",
                config=self.config,
                name=f"{experiment_name}_{model_name}",
                tags=[
                    f"model_{model_name}",
                    f"optimizer_{optimizer_name}",
                    f"scheduler_{scheduler_name}"
                ]   
            )
            self.logger.info("WandB initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
            self.run = None
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to WandB and console"""
        if self.run:
            wandb.log(metrics, step=step)
        
        # Log to console
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    
    def log_model_info(self, model, input_shape):
        """Log model architecture information"""
        if self.run:
            try:
                # Count parameters
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                
                model_info = {
                    "model_total_params": total_params,
                    "model_trainable_params": trainable_params,
                    "model_input_shape": list(input_shape)
                }
                
                wandb.log(model_info)
                self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
                
            except Exception as e:
                self.logger.warning(f"Failed to log model info: {e}")
    
    def watch_model(self, model, log_freq: int = 100):
        """Watch model gradients and parameters"""
        if self.run:
            try:
                wandb.watch(model, log="all", log_freq=log_freq)
            except Exception as e:
                self.logger.warning(f"Failed to watch model: {e}")
    
    def log_hyperparameters(self, hparams: Dict[str, Any]):
        """Log hyperparameters"""
        if self.run:
            wandb.config.update(hparams, allow_val_change=True)

        self.logger.info(f"Hyperparameters: {hparams}")
    
    def save_model_artifact(self, model_path: str, name: str = "model"):
        """Save model as artifact"""
        if self.run and os.path.exists(model_path):
            try:
                artifact = wandb.Artifact(name, type="model")
                artifact.add_file(model_path)
                self.run.log_artifact(artifact)
                self.logger.info(f"Model artifact saved: {model_path}")
            except Exception as e:
                self.logger.warning(f"Failed to save model artifact: {e}")
    
    def finish(self):
        """Finish WandB logging"""
        if self.run:
            wandb.finish()
            self.logger.info("WandB logging finished")

def setup_logger(config: Dict[str, Any]) -> WandbLogger:
    """Setup logger with configuration"""
    project_name = config.get('logging', {}).get('project_name', 'deep_pro_experiments')
    return WandbLogger(config, project_name)