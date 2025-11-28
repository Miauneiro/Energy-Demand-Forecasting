# Energy Demand Forecasting Platform

Professional electricity demand forecasting system using deep learning.

## Overview

This platform provides 24-hour electricity demand forecasts using a 4-layer LSTM neural network with 1.86M parameters. The system achieves < 2.5% MAPE on test data and is optimized for real-world energy sector applications.

## Features

- **Multiple Forecast Horizons**: 24-hour, 1-week, and 1-month forecasts
- **Business Context**: Guided selection based on use case and user role
- **Interactive Visualizations**: Plotly-based charts with confidence intervals
- **Professional UI**: Clean, LaTeX-enabled interface for technical users
- **Production Ready**: Deployable to Streamlit Cloud

## Model Architecture

```
Input: (batch, 168, 13) - 1 week of 13 temporal features
  ↓
LSTM Layer 1: 256 hidden units
  ↓
LSTM Layer 2: 256 hidden units
  ↓
LSTM Layer 3: 256 hidden units
  ↓
LSTM Layer 4: 256 hidden units
  ↓
Output: (batch, 24, 1) - 24-hour forecast
```

**Total Parameters**: 1,862,680  
**Training Time**: ~4 hours (NVIDIA GTX 1050)  
**Target Accuracy**: < 2.5% MAPE

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/energy-forecasting.git
cd energy-forecasting

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run app.py
```

## Usage

### Local Development

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser.

### Deployment

The app is ready for deployment to Streamlit Cloud:

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Deploy with one click

## Project Structure

```
energy-forecasting/
├── app.py                  # Main Streamlit application
├── src/
│   ├── model.py           # LSTM model architecture
│   ├── preprocessing.py   # Data preprocessing pipeline
│   └── train.py          # Training script
├── models/
│   └── best_model.pth    # Trained model weights
├── data/
│   ├── raw/              # Raw data files
│   └── processed/        # Processed data & scalers
├── plots/                # Training curves
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Model Performance

| Metric | Value |
|--------|-------|
| Test MAPE | < 2.5% |
| Test MAE | < 12,000 MW |
| Train Loss | < 0.001 |
| Val Loss | < 0.002 |

## Use Cases

### 24-Hour Forecast
- Day-ahead energy market trading
- Real-time grid balancing
- Operational staffing decisions
- Spot price optimization

### 1-Week Forecast
- Generator unit commitment
- Maintenance scheduling
- Fuel procurement planning
- Medium-term capacity allocation

### 1-Month Forecast
- Long-term capacity planning
- Infrastructure investment decisions
- Seasonal resource allocation
- Budget forecasting

## Technical Specifications

**Framework**: PyTorch 2.1.0  
**Frontend**: Streamlit 1.29.0  
**Visualization**: Plotly 5.17.0  
**Hardware**: NVIDIA GTX 1050 (2GB VRAM)  
**Training**: 150 epochs, batch size 256  
**Optimization**: Adam with ReduceLROnPlateau scheduler

## Author

**João Manero**  
Meteorology Student | Aspiring Data Scientist  
Universidade de Lisboa

## License

MIT License - see LICENSE file for details

## Acknowledgments

- Built as part of portfolio development for data science roles
- Optimized for production deployment and stakeholder communication
- Designed with business value and user experience in mind
