import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
from typing import Tuple, Dict, Any

def get_transforms(config: Dict[str, Any]) -> Tuple[transforms.Compose, transforms.Compose]:
    """Get training and validation transforms (no augmentation)"""
    
    # Get preprocessing config (normalization only)
    prep_config = config.get('data_preprocessing', {})
    
    # Both training and validation use same transforms (no augmentation)
    base_transforms = [transforms.ToTensor()]
    
    # Add normalization if specified
    if 'normalize' in prep_config:
        norm_config = prep_config['normalize']
        base_transforms.append(
            transforms.Normalize(
                mean=norm_config.get('mean', [0.485, 0.456, 0.406]),
                std=norm_config.get('std', [0.229, 0.224, 0.225])
            )
        )
    
    transform_train = transforms.Compose(base_transforms)
    transform_val = transforms.Compose(base_transforms)
    
    return transform_train, transform_val

def get_data_loaders(config: Dict[str, Any], data_dir: str) -> Tuple[DataLoader, DataLoader]:
    """Get training and validation data loaders"""
    
    data_config = config.get('data', {})
    
    # Get transforms
    transform_train, transform_val = get_transforms(config)
    
    # Create datasets
    trainset = ImageFolder(
        os.path.join(data_dir, "train"), 
        transform=transform_train
    )
    valset = ImageFolder(
        os.path.join(data_dir, "val"), 
        transform=transform_val
    )
    
    # Create data loaders
    trainloader = DataLoader(
        trainset,
        batch_size=data_config.get('batch_size', 256),
        shuffle=True,
        num_workers=data_config.get('num_workers', 6),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    valloader = DataLoader(
        valset,
        batch_size=data_config.get('val_batch_size', 64),
        shuffle=False,
        num_workers=data_config.get('num_workers', 6),
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return trainloader, valloader

def setup_data_directory(url: str, root: str = "./content/data") -> str:
    """Setup data directory by downloading and extracting dataset"""
    from torchvision.datasets.utils import download_url, extract_archive
    
    zip_path = os.path.join(root, "project_dataset.zip")
    out_dir = os.path.join(root, "project_dataset")
    
    os.makedirs(root, exist_ok=True)
    
    # Download if not exists
    if not os.path.exists(zip_path):
        download_url(url, root=root, filename="project_dataset.zip")
    
    # Extract if not exists
    if not os.path.exists(out_dir):
        extract_archive(zip_path, out_dir)
    
    return os.path.join(out_dir, "data/ProjectDataset")