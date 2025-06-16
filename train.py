import os
import random
import numpy as np
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from tqdm import tqdm

from utils.logger import setup_logger
from utils.data_utils import setup_data_directory, get_data_loaders
from utils.model_manager import ModelManager

# URL for dataset download
URL = "https://github.com/JanghunHyeon/AISW4202-Project/releases/download/v.1.1.0/project_dataset.zip"

def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def create_model(config: DictConfig):
    """Create model from config"""
    return hydra.utils.instantiate(config.model)

def create_optimizer(model, config: DictConfig):
    """Create optimizer from config"""
    return hydra.utils.instantiate(config.optimizer, model.parameters())

def create_scheduler(optimizer, config: DictConfig):
    """Create scheduler from config"""
    return hydra.utils.instantiate(config.scheduler, optimizer)

def create_criterion(config: DictConfig):
    """Create loss criterion"""
    criterion_name = config.training.get('criterion', 'crossentropy').lower()
    if criterion_name == 'crossentropy':
        return nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported criterion: {criterion_name}")

def train_one_epoch(model, trainloader, criterion, optimizer, device, logger, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(trainloader, desc=f'Epoch {epoch+1} Train')

    step = 0
    for i, (inputs, labels) in enumerate(pbar):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if i % 50 == 49:
            avg_loss = running_loss / 50
            accuracy = 100 * correct / total
            pbar.set_postfix({
                'Loss': f'{avg_loss:.3f}',
                'Acc': f'{accuracy:.2f}%'
            })
            
            # Log metrics
            logger.log_metrics({
                'train_loss': avg_loss,
                'train_accuracy': accuracy
            }, step=epoch * len(trainloader) + i)
            
            running_loss = 0.0
        
        step += 1
    
    final_accuracy = 100 * correct / total
    return final_accuracy

def validate(model, valloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(valloader, desc='Validation'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    val_loss = running_loss / len(valloader)
    val_accuracy = 100 * correct / total
    
    return val_loss, val_accuracy

def save_model(model, save_path: str, team_idx: str = "team1"):
    """Save model using TorchScript (legacy function)"""
    os.makedirs(save_path, exist_ok=True)
    
    model.eval()
    example_input = torch.randn(1, 3, 48, 48)
    if torch.cuda.is_available():
        example_input = example_input.cuda()
        
    traced_script = torch.jit.trace(model, example_input)
    fname = os.path.join(save_path, f"model_{team_idx}.pt")
    traced_script.save(fname)
    
    return fname

@hydra.main(config_path="configs", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Set up experiment
    set_seed(cfg.experiment.seed, cfg.training.deterministic)
    device = torch.device(cfg.experiment.device if torch.cuda.is_available() else 'cpu')
    
    # Set up logging
    logger = setup_logger(cfg)
    logger.logger.info(f"Starting experiment: {cfg.experiment.name}")
    logger.logger.info(f"Using device: {device}")
    
    # Setup data
    data_dir = setup_data_directory(URL)
    trainloader, valloader = get_data_loaders(cfg, data_dir)
    logger.logger.info(f"Training batches: {len(trainloader)}, Validation batches: {len(valloader)}")
    
    # Create model
    model = create_model(cfg).to(device)
    logger.log_model_info(model, (1, 3, cfg.data.input_size, cfg.data.input_size))
    logger.watch_model(model, log_freq=cfg.logging.get('log_freq', 100))
    
    # Create optimizer, scheduler, and criterion
    optimizer = create_optimizer(model, cfg)
    scheduler = create_scheduler(optimizer, cfg)
    criterion = create_criterion(cfg).to(device)
    
    # Log hyperparameters
    hparams = {
        'model': cfg.model._target_.split('.')[-1],
        'optimizer': cfg.optimizer._target_.split('.')[-1],
        'scheduler': cfg.scheduler._target_.split('.')[-1],
        'learning_rate': cfg.optimizer.lr,
        'batch_size': cfg.data.batch_size,
        'num_epochs': cfg.experiment.num_epochs
    }
    logger.log_hyperparameters(hparams)
    
    # Initialize model manager
    model_manager = ModelManager()
    model_name = cfg.model._target_.split('.')[-1]
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Convert config to dict for saving
    config_dict = {
        'model_config': dict(cfg.model),
        'optimizer_config': dict(cfg.optimizer),
        'scheduler_config': dict(cfg.scheduler),
        'data_config': dict(cfg.data),
        'experiment_config': dict(cfg.experiment)
    }
    
    # Training loop
    best_val_accuracy = 0.0
    best_model_path = None
    
    for epoch in range(cfg.experiment.num_epochs):
        # Train
        train_accuracy = train_one_epoch(
            model, trainloader, criterion, optimizer, device, logger, epoch
        )
        
        # Validate
        val_loss, val_accuracy = validate(model, valloader, criterion, device)
        
        # Step scheduler
        scheduler.step()
        
        # Log epoch metrics with proper step
        current_step = (epoch + 1) * len(trainloader)
        logger.log_metrics({
            'epoch': epoch + 1,
            'train_accuracy_epoch': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=current_step)
        
        logger.logger.info(
            f'Epoch [{epoch+1}/{cfg.experiment.num_epochs}] '
            f'Train Acc: {train_accuracy:.3f}% | '
            f'Val Loss: {val_loss:.3f} | '
            f'Val Acc: {val_accuracy:.3f}%'
        )
        
        # Save best model using model manager
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            if cfg.model_save.save_best:
                best_model_path = model_manager.save_model_checkpoint(
                    model=model,
                    model_name=model_name,
                    accuracy=val_accuracy,
                    epoch=epoch,
                    is_best=True,
                    timestamp=timestamp,
                    config=config_dict
                )
                logger.logger.info(f'New best model saved: {best_model_path} (Val Acc: {val_accuracy:.3f}%)')
    
    # Final evaluation
    _, final_val_accuracy = validate(model, valloader, criterion, device)
    logger.logger.info(f'Final validation accuracy: {final_val_accuracy:.3f}%')
    
    # Save final model using model manager
    if cfg.model_save.save_last:
        final_model_path = model_manager.save_model_checkpoint(
            model=model,
            model_name=model_name,
            accuracy=final_val_accuracy,
            epoch=cfg.experiment.num_epochs-1,
            is_best=False,
            timestamp=timestamp,
            config=config_dict
        )
        logger.logger.info(f'Final model saved: {final_model_path}')
        
        # Save as artifact
        logger.save_model_artifact(final_model_path, "final_model")
        if best_model_path:
            logger.save_model_artifact(best_model_path, "best_model")
    
    # Show model summary
    print(f"\n=== Experiment Summary ===")
    print(f"Model: {model_name}")
    print(f"Best Validation Accuracy: {best_val_accuracy:.3f}%")
    
    # Show global best info
    global_best = model_manager.get_global_best_info()
    if global_best:
        print(f"\n=== Global Best Model ===")
        print(f"Model: {global_best['model_name']}")
        print(f"Accuracy: {global_best['accuracy']:.3f}%")
        print(f"Path: {global_best['model_path']}")
    
    # Finish logging
    logger.finish()
    
    print(f'\nTraining completed! Best validation accuracy: {best_val_accuracy:.3f}%')

if __name__ == "__main__":
    main()