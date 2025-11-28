"""
LSTM Forecasting Model for Electricity Demand Prediction.

Optimized for:
- Input: (batch, 168, 13) - 1 week of 13 features
- Output: (batch, 24, 1) - 1 day ahead forecast
- Hardware: GTX 1050 2GB VRAM (conservative settings)
"""

# Fix DLL loading for Windows
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
            critical_dlls = ['asmjit.dll', 'libiomp5md.dll', 'fbgemm.dll']
            for dll_name in critical_dlls:
                dll_path = os.path.join(torch_lib_path, dll_name)
                if os.path.exists(dll_path):
                    try:
                        ctypes.CDLL(dll_path)
                    except:
                        pass

import torch
import torch.nn as nn
from typing import Tuple


class LSTMForecaster(nn.Module):
    """
    LSTM-based electricity demand forecaster.
    
    Architecture:
    - Multi-layer LSTM for sequence processing
    - Dropout for regularization
    - Fully connected layer for output projection
    
    Args:
        input_size: Number of input features (default: 13)
        hidden_size: LSTM hidden state size (default: 64, conservative for 2GB VRAM)
        num_layers: Number of stacked LSTM layers (default: 2)
        output_size: Forecast horizon (default: 24 hours)
        dropout: Dropout probability (default: 0.2)
    """
    
    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 24,
        dropout: float = 0.2
    ):
        super(LSTMForecaster, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input: (batch, seq, features)
            dropout=dropout if num_layers > 1 else 0,  # Dropout between LSTM layers
            bidirectional=False  # Unidirectional for time series
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer to project to output size
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases
                param.data.fill_(0)
                # Set forget gate bias to 1 (helps learning)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
            elif 'fc' in name and 'weight' in name:
                # Fully connected weights
                nn.init.xavier_uniform_(param.data)
    
    def forward(
        self, 
        x: torch.Tensor,
        hidden: Tuple[torch.Tensor, torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, lookback, features)
            hidden: Optional initial hidden state
            
        Returns:
            Output tensor of shape (batch, horizon, 1)
        """
        batch_size = x.size(0)
        
        # LSTM forward pass
        # lstm_out: (batch, lookback, hidden_size)
        # hidden: (h_n, c_n) each of shape (num_layers, batch, hidden_size)
        if hidden is None:
            lstm_out, hidden = self.lstm(x)
        else:
            lstm_out, hidden = self.lstm(x, hidden)
        
        # Take the last output of the sequence
        # last_output: (batch, hidden_size)
        last_output = lstm_out[:, -1, :]
        
        # Apply dropout
        last_output = self.dropout(last_output)
        
        # Project to output size
        # output: (batch, output_size)
        output = self.fc(last_output)
        
        # Reshape to (batch, output_size, 1) for consistency with target shape
        output = output.unsqueeze(-1)
        
        return output
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LSTMForecasterLarge(nn.Module):
    """
    Larger LSTM model for better performance (if you have more VRAM or use CPU).
    
    Architecture:
    - Deeper LSTM (3 layers)
    - Larger hidden size (128)
    - More capacity for complex patterns
    
    Use this if:
    - Training on CPU (no VRAM limit)
    - You have >4GB VRAM
    - You want better accuracy
    """
    
    def __init__(
        self,
        input_size: int = 13,
        hidden_size: int = 128,
        num_layers: int = 3,
        output_size: int = 24,
        dropout: float = 0.3
    ):
        super(LSTMForecasterLarge, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=False
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n//4:n//2].fill_(1)
            elif 'fc' in name and 'weight' in name:
                nn.init.xavier_uniform_(param.data)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.dropout(last_output)
        output = self.fc(last_output)
        output = output.unsqueeze(-1)
        return output
    
    def get_num_params(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    model_size: str = 'small',
    input_size: int = 13,
    output_size: int = 24,
    device: str = None,
    custom_config: dict = None
) -> nn.Module:
    """
    Factory function to create LSTM model.
    
    Args:
        model_size: 'small' (safe), 'large' (powerful), 'beast' (max GPU usage)
        input_size: Number of input features
        output_size: Forecast horizon
        device: Device to place model on (auto-detect if None)
        custom_config: Dict with custom hyperparameters
        
    Returns:
        LSTM model on specified device
    """
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    if model_size == 'small':
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=64,
            num_layers=2,
            output_size=output_size,
            dropout=0.2
        )
    elif model_size == 'large':
        model = LSTMForecasterLarge(
            input_size=input_size,
            hidden_size=128,
            num_layers=3,
            output_size=output_size,
            dropout=0.3
        )
    elif model_size == 'beast':
        # BEAST MODE: Max out that GTX 1050!
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=96,      # Increased from 64
            num_layers=3,        # Increased from 2
            output_size=output_size,
            dropout=0.3
        )
        print("🔥 BEAST MODE ACTIVATED!")
        print("   Hidden size: 96 (vs 64)")
        print("   Layers: 3 (vs 2)")
        print("   Use with batch_size=32-48")
    elif custom_config:
        # Custom configuration
        model = LSTMForecaster(
            input_size=input_size,
            hidden_size=custom_config.get('hidden_size', 64),
            num_layers=custom_config.get('num_layers', 2),
            output_size=output_size,
            dropout=custom_config.get('dropout', 0.2)
        )
    else:
        raise ValueError(f"Unknown model_size: {model_size}. Use 'small', 'large', or 'beast'")
    
    # Move to device
    model = model.to(device)
    
    # Print model info
    num_params = model.get_num_params()
    print(f"📊 Model: {model.__class__.__name__}")
    print(f"   Device: {device}")
    print(f"   Parameters: {num_params:,}")
    print(f"   Input shape: (batch, 168, {input_size})")
    print(f"   Output shape: (batch, {output_size}, 1)")
    
    return model


def model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (32, 168, 13)):
    """
    Print detailed model summary.
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch, seq_len, features)
    """
    print("=" * 70)
    print("  MODEL ARCHITECTURE")
    print("=" * 70)
    print(model)
    print("\n" + "=" * 70)
    print("  LAYER DETAILS")
    print("=" * 70)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
        
        print(f"{name:30s} | Shape: {str(list(param.shape)):20s} | Params: {param_count:,}")
    
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    
    # Estimate memory usage
    param_size = total_params * 4 / (1024**2)  # 4 bytes per float32, convert to MB
    print(f"\nEstimated model size: {param_size:.2f} MB")
    
    # Test forward pass
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_size).to(device)
    
    try:
        with torch.no_grad():
            output = model(dummy_input)
        print(f"\n✅ Forward pass successful!")
        print(f"   Input shape: {tuple(dummy_input.shape)}")
        print(f"   Output shape: {tuple(output.shape)}")
    except Exception as e:
        print(f"\n❌ Forward pass failed: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    print("\n🏗️  LSTM Model Builder\n")
    
    # Create small model (for GTX 1050)
    print("Creating SMALL model (optimized for GTX 1050 2GB VRAM)...")
    model_small = create_model(model_size='small')
    print()
    
    # Test it
    print("Testing model...")
    device = next(model_small.parameters()).device
    dummy_input = torch.randn(32, 168, 13).to(device)
    
    with torch.no_grad():
        output = model_small(dummy_input)
    
    print(f"✅ Test passed!")
    print(f"   Input: {tuple(dummy_input.shape)}")
    print(f"   Output: {tuple(output.shape)}")
    print()
    
    # Show detailed summary
    model_summary(model_small)
    
    print("\n" + "=" * 70)
    print("  READY FOR TRAINING!")
    print("=" * 70)
    print("\n💡 Usage in training:")
    print("""
    from model import create_model
    
    # Create model
    model = create_model(model_size='small', device='cuda')
    
    # Forward pass
    predictions = model(input_batch)  # (batch, 24, 1)
    
    # Calculate loss
    loss = criterion(predictions, targets)
    """)
