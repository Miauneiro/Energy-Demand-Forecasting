"""
Training Script for LSTM Energy Demand Forecaster.

MEGA BEAST Configuration:
- Hidden: 256, Layers: 4, Batch: 256
- Parameters: 1.86M
- VRAM: 1488 MB (73%)
- Training time: ~18 minutes (150 epochs)
- Target MAPE: < 2.5%
"""

# Fix DLL loading
import sys
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

if sys.platform == 'win32' and sys.version_info >= (3, 8):
    import importlib.util
    import ctypes
    torch_spec = importlib.util.find_spec('torch')
    if torch_spec and torch_spec.origin:
        torch_lib_path = os.path.join(os.path.dirname(torch_spec.origin), 'lib')
        if os.path.exists(torch_lib_path):
            os.add_dll_directory(torch_lib_path)
            for dll_name in ['asmjit.dll', 'libiomp5md.dll', 'fbgemm.dll']:
                dll_path = os.path.join(torch_lib_path, dll_name)
                if os.path.exists(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except:
                        pass

from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time

from preprocessing import quick_preprocess
from model import LSTMForecaster


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience=15, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'   EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


class Trainer:
    """Complete training pipeline for LSTM forecaster."""
    
    def __init__(self, config):
        self.config = config
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create directories
        Path(config['checkpoint_dir']).mkdir(parents=True, exist_ok=True)
        Path(config['plot_dir']).mkdir(parents=True, exist_ok=True)
        
        # History tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
            'epoch_time': []
        }
        
    def train(self):
        """Main training loop."""
        
        print("\n" + "="*70)
        print("  🔥 MEGA BEAST TRAINING - MAXIMUM ACCURACY MODE 🔥")
        print("="*70)
        
        # 1. Load data
        print("\n📦 Loading and preprocessing data...")
        train_loader, val_loader, test_loader, preprocessor = quick_preprocess(
            filepath=self.config['data_path'],
            lookback=self.config['lookback'],
            horizon=self.config['horizon'],
            batch_size=self.config['batch_size']
        )
        
        self.preprocessor = preprocessor
        self.test_loader = test_loader
        
        # 2. Create model
        print("\n🏗️  Creating MEGA BEAST model...")
        model = LSTMForecaster(
            input_size=self.config['input_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            output_size=self.config['output_size'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
        print(f"   Device: {self.device}")
        
        if self.device == 'cuda':
            print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
        # 3. Setup training
        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=self.config['scheduler_factor'],
            patience=self.config['scheduler_patience'],
            min_lr=self.config['min_lr']
        )
        
        early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience'],
            verbose=True
        )
        
        # 4. Training loop
        print("\n" + "="*70)
        print("  🚀 STARTING TRAINING")
        print("="*70)
        print(f"\n   Epochs: {self.config['epochs']}")
        print(f"   Batch size: {self.config['batch_size']}")
        print(f"   Train batches: {len(train_loader)}")
        print(f"   Val batches: {len(val_loader)}")
        print()
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(1, self.config['epochs'] + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self._train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss = self._validate_epoch(model, val_loader, criterion)
            
            # Scheduler step
            scheduler.step(val_loss)
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['lr'].append(optimizer.param_groups[0]['lr'])
            
            epoch_time = time.time() - epoch_start
            self.history['epoch_time'].append(epoch_time)
            
            # Print progress
            print(f"Epoch {epoch:3d}/{self.config['epochs']} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self._save_checkpoint(model, optimizer, epoch, val_loss, is_best=True)
                print(f"   💾 Best model saved! (Val Loss: {val_loss:.6f})")
            
            # Early stopping
            early_stopping(val_loss, epoch)
            if early_stopping.early_stop:
                print(f"\n⏹️  Early stopping triggered at epoch {epoch}")
                print(f"   Best epoch: {early_stopping.best_epoch}")
                print(f"   Best val loss: {early_stopping.best_loss:.6f}")
                break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(model, optimizer, epoch, val_loss, is_best=False)
        
        total_time = time.time() - start_time
        
        print("\n" + "="*70)
        print("  ✅ TRAINING COMPLETE")
        print("="*70)
        print(f"\n   Total time: {total_time/60:.1f} minutes")
        print(f"   Best val loss: {best_val_loss:.6f}")
        print(f"   Best epoch: {early_stopping.best_epoch}")
        print(f"   Final LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # 5. Plot training curves
        self._plot_training_curves()
        
        # 6. Save training history
        self._save_history()
        
        return model
    
    def _train_epoch(self, model, train_loader, criterion, optimizer):
        """Train for one epoch."""
        model.train()
        total_loss = 0
        
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            
            # Forward pass
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config['gradient_clip']
            )
            
            optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, model, val_loader, criterion):
        """Validate for one epoch."""
        model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def _save_checkpoint(self, model, optimizer, epoch, val_loss, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config,
            'history': self.history
        }
        
        if is_best:
            path = Path(self.config['checkpoint_dir']) / 'best_model.pth'
        else:
            path = Path(self.config['checkpoint_dir']) / f'checkpoint_epoch_{epoch}.pth'
        
        torch.save(checkpoint, path)
    
    def _plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss curves
        epochs = range(1, len(self.history['train_loss']) + 1)
        axes[0].plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Learning rate
        axes[1].plot(epochs, self.history['lr'], 'g-', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Learning Rate', fontsize=12)
        axes[1].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        axes[1].set_yscale('log')
        axes[1].grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = Path(self.config['plot_dir']) / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"\n📊 Training curves saved to: {plot_path}")
        
        plt.close()
    
    def _save_history(self):
        """Save training history as JSON."""
        history_path = Path(self.config['checkpoint_dir']) / 'training_history.json'
        
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        print(f"📝 Training history saved to: {history_path}")


def main():
    """Main training function."""
    
    # MEGA BEAST Configuration
    config = {
        # Data
        'data_path': 'data/raw/us_demand_historical.csv',
        'lookback': 168,
        'horizon': 24,
        'batch_size': 256,  # 🔥 MEGA BEAST
        
        # Model
        'input_size': 13,
        'hidden_size': 256,  # 🔥 MEGA BEAST
        'num_layers': 4,     # 🔥 MEGA BEAST
        'output_size': 24,
        'dropout': 0.3,
        
        # Training
        'epochs': 150,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'gradient_clip': 1.0,
        
        # Scheduler
        'scheduler_factor': 0.5,
        'scheduler_patience': 5,
        'min_lr': 1e-6,
        
        # Early stopping
        'early_stopping_patience': 20,
        
        # Saving
        'checkpoint_dir': 'models',
        'plot_dir': 'plots',
    }
    
    print("\n🔥 MEGA BEAST Configuration:")
    print(f"   Model: {config['hidden_size']} hidden, {config['num_layers']} layers")
    print(f"   Batch size: {config['batch_size']}")
    print(f"   Parameters: ~1.86M")
    print(f"   Expected VRAM: 1488 MB (73%)")
    print(f"   Expected time: ~18 minutes (150 epochs)")
    print(f"   Target MAPE: < 2.5%")
    
    # Train
    trainer = Trainer(config)
    model = trainer.train()
    
    print("\n" + "="*70)
    print("  🎉 MODEL TRAINING COMPLETE!")
    print("="*70)
    print("\n📁 Outputs:")
    print(f"   Best model: models/best_model.pth")
    print(f"   Training curves: plots/training_curves.png")
    print(f"   Training history: models/training_history.json")
    
    print("\n🎯 Next steps:")
    print("   1. Evaluate on test set (notebooks/04_evaluation.ipynb)")
    print("   2. Calculate metrics (MAE, RMSE, MAPE)")
    print("   3. Visualize predictions")
    print("   4. Deploy to Streamlit!")
    
    return model


if __name__ == "__main__":
    model = main()
