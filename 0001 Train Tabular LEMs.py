"""
Train Tabular LEMs (Large Events Models)

This script trains various neural network architectures on soccer event data in LEM format.
It supports both survey-style quick training and full training modes.
The models include MLPs of various sizes, with configurable hyperparameters.

The script can be run in two modes:
1. Survey mode: Quick training to compare different architectures
2. Full mode: Complete training of selected architectures
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from typing import List, Dict, Optional, Union
from datetime import datetime
from pathlib import Path

# Constants for CUDA setup
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MLP(nn.Module):
    """Multi-layer Perceptron with configurable architecture."""
    
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int, dropout_rate: float = 0.0):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def write_to_tracker(filepath: str, line: str) -> None:
    """Write a line to the tracking file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as f:
        f.write(line)

def instantiate_models(seq_len: int, output_size: int, mode: str = 'survey') -> List[nn.Module]:
    """
    Instantiate models based on the specified mode.
    
    Args:
        seq_len: Length of input sequence
        output_size: Number of output classes
        mode: Either 'survey' for quick comparison or 'full' for complete training
    
    Returns:
        List of instantiated PyTorch models
    """
    models = []
    
    if mode == 'survey':
        # Survey mode includes a wider range of architectures
        models.extend([
            MLP(seq_len, [80], output_size),
            MLP(seq_len, [96, 96, 96], output_size),
            MLP(seq_len, [196, 196, 196], output_size),
            MLP(seq_len, [360, 360, 360], output_size),
            MLP(seq_len, [682, 682, 682], output_size),
            MLP(seq_len, [1200, 1200, 1200], output_size),
            MLP(seq_len, [2220, 2220, 2220], output_size, dropout_rate=0.3)
        ])
    else:
        # Full mode focuses on selected architectures with dropout
        models.extend([
            MLP(seq_len, [196, 196, 196], output_size, dropout_rate=0.3),
            MLP(seq_len, [360, 360, 360], output_size, dropout_rate=0.3),
            MLP(seq_len, [682, 682, 682], output_size, dropout_rate=0.3),
            MLP(seq_len, [1200, 1200, 1200], output_size, dropout_rate=0.3),
            MLP(seq_len, [2220, 2220, 2220], output_size, dropout_rate=0.3)
        ])
    
    return models

def load_data(data_path: str, val_samples: int = 100_000) -> tuple:
    """
    Load and prepare validation data.
    
    Args:
        data_path: Path to validation data file
        val_samples: Number of validation samples to use
    
    Returns:
        Tuple of (validation dataset, output size)
    """
    val_data = pd.read_feather(data_path)
    val_data = val_data.sample(val_samples, random_state=42)
    
    X_val = torch.tensor(val_data.drop(columns=['target']).astype(int).values, dtype=torch.float32).to(DEVICE)
    Y_val = torch.tensor(pd.get_dummies(val_data['target']).astype(int).values, dtype=torch.float32).to(DEVICE)
    
    val_dataset = TensorDataset(X_val, Y_val)
    output_size = Y_val.shape[1]
    
    return val_dataset, output_size

def train_models(
    seq_len: int,
    mode: str,
    data_dir: str,
    output_dir: str,
    learning_rate: float,
    batch_size: int,
    n_epochs: int,
    n_files_train: int,
    checkpoints: List[int],
    val_samples: int = 100_000
) -> None:
    """
    Train models on the specified data.
    
    Args:
        seq_len: Length of input sequence
        mode: Either 'survey' for quick comparison or 'full' for complete training
        data_dir: Directory containing the data files
        output_dir: Directory to save model outputs
        learning_rate: Learning rate for optimization
        batch_size: Batch size for training
        n_epochs: Number of epochs to train
        n_files_train: Number of training files to use
        checkpoints: List of batch numbers at which to evaluate
        val_samples: Number of validation samples to use
    """
    # Load validation data
    val_data_path = os.path.join(data_dir, f'val_extensive_2223_sq{seq_len}_rs42_0.feather')
    val_dataset, output_size = load_data(val_data_path, val_samples)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*4, shuffle=False, drop_last=False)
    
    # Initialize models
    models = instantiate_models(seq_len, output_size, mode)
    
    # Training loop
    for model in models:
        model_name = model.__class__.__name__
        model_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Training {model_name} with {model_params} parameters")
        
        model = model.to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        best_loss = float('inf')
        batch_counter = 0
        
        for epoch in range(n_epochs):
            for train_set_id in tqdm(range(n_files_train), desc=f"Epoch {epoch + 1}/{n_epochs}"):
                # Load training data
                train_data_path = os.path.join(
                    data_dir,
                    f'train_extensive_1516_2122_sq{seq_len}_rs42_{train_set_id}.feather'
                )
                train_data = pd.read_feather(train_data_path)
                
                X_train = torch.tensor(train_data.drop(columns=['target']).astype(int).values, dtype=torch.float32)
                Y_train = torch.tensor(pd.get_dummies(train_data['target']).astype(int).values, dtype=torch.float32)
                
                train_dataset = TensorDataset(X_train, Y_train)
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                
                # Training
                model.train()
                train_losses = []
                
                for X_batch, Y_batch in train_loader:
                    X_batch, Y_batch = X_batch.to(DEVICE), Y_batch.to(DEVICE)
                    
                    optimizer.zero_grad()
                    Y_pred = model(X_batch)
                    loss = criterion(Y_pred, Y_batch)
                    loss.backward()
                    optimizer.step()
                    
                    train_losses.append(loss.item())
                    batch_counter += 1
                    
                    if batch_counter in checkpoints:
                        # Validation
                        model.eval()
                        with torch.no_grad():
                            val_losses = []
                            for X_val, Y_val in val_loader:
                                Y_pred = model(X_val)
                                val_loss = criterion(Y_pred, Y_val)
                                val_losses.append(val_loss.item())
                            
                            val_loss = np.mean(val_losses)
                            train_loss = np.mean(train_losses)
                            
                            # Log results
                            tracker_path = os.path.join(output_dir, f'trackers/71{"12" if mode == "full" else "11"}_definitive.csv')
                            log_line = f'{model_name},{seq_len},{model_params},{epoch},{batch_counter},{train_loss},{val_loss},{datetime.now()}\\n'
                            write_to_tracker(tracker_path, log_line)
                            
                            # Save best model
                            if val_loss < best_loss and mode == 'full':
                                best_loss = val_loss
                                model_path = os.path.join(
                                    output_dir,
                                    f'models/7112_{model_name}_{model_params}_{seq_len}_e{epoch}.pt'
                                )
                                torch.save(model.state_dict(), model_path)
                        
                        model.train()
                        train_losses = []

def main():
    """Main function to run the training pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Tabular LEMs")
    parser.add_argument('--mode', type=str, choices=['survey', 'full'], required=True,
                      help="Training mode: 'survey' for quick comparison or 'full' for complete training")
    parser.add_argument('--data_dir', type=str, required=True,
                      help="Directory containing the data files")
    parser.add_argument('--output_dir', type=str, required=True,
                      help="Directory to save model outputs")
    parser.add_argument('--seq_lengths', type=int, nargs='+', default=[1, 3, 5, 7, 9],
                      help="Sequence lengths to process")
    
    args = parser.parse_args()
    
    # Set hyperparameters based on mode
    if args.mode == 'survey':
        config = {
            'learning_rate': 0.01,
            'batch_size': 1024,
            'n_epochs': 1,
            'n_files_train': 8,
            'checkpoints': sorted([10 ** i for i in range(2, 10)] + [3 * (10 ** i) for i in range(2, 10)])
        }
    else:
        config = {
            'learning_rate': 0.001,
            'batch_size': 1024,
            'n_epochs': 4,
            'n_files_train': 30,
            'checkpoints': [i * (10 ** 4) for i in range(1, 100)]
        }
    
    # Create output directories
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'trackers'), exist_ok=True)
    
    # Train models for each sequence length
    for seq_len in args.seq_lengths:
        print(f"Training models for sequence length {seq_len}")
        train_models(seq_len, args.mode, args.data_dir, args.output_dir, **config)

if __name__ == "__main__":
    main() 