# Economic Data Processing and Prediction

This repository contains the complete codebase for the bachelor's thesis "Economic Data Processing and Prediction". The thesis focuses on time series forecasting using neural networks for economic data from the World Bank.

## Project Overview

This project attempts to create a pipeline for automated download, preprocessing, and modelling of economical indicators.

- **Data Processing**: Comprehensive preprocessing pipeline for World Bank economic indicators
- **Models**: Implementation and comparison of LSTM, GRU, Informer, and SpaceTimeFormer architectures
- **Experiments**: Systematic evaluation of preprocessing strategies and model performance

## Repository Structure

```
├── modelling/             # Neural network models and training
│   ├── core/              # Core ML infrastructure
│   ├── models/            # Model implementations
│   ├── configs/           # Model configuration files (intended)
│   └── tuning/            # Hyperparameter tuning utilities (intended)
├── raw_data/              # Data loading and management
│   ├── _core/             # Core data infrastructure
│   ├── dataset.py         # Main dataset class
│   ├── wb_api.py          # World Bank API interface
│   ├── core.py            # Controller class orchestrating the download/save/load process
│   └── ...                # Additional data utilities
├── preprocessing/         # Data preprocessing pipeline
│   ├── preprocess.py      # Main preprocessing classes
│   ├── gain.py            # GAIN imputation implementation
│   └── ...                # Preprocessing utilities
├── config/                # Experiment configurations
│   └── template.yaml      # Configuration template
├── scripts/               # Utility scripts
├── runs/                  # Experiment outputs (generated)
```

## Quick Start

### Prerequisites

- Python 3.11+

### Installation

```bash
# Clone the repository
git clone <repository-url>

# Create virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Recreating Experiments

```bash
# Recreate the datasets
python thesis_datasets.py

# Recreate experiments from the thesis
python quick_block_runner.py -a  # Experiment A
python quick_block_runner.py -b  # Experiment B  
python quick_block_runner.py -c  # Experiment C
```

#### Notes on the implementation

The originally intended, proper, way to run experiments is `run_experiments_block.py`, coupled with experiment config `.yaml` files. However, due to time-constraints and rapid alteration of the experiments used in the thesis, this experimental framework hasn't been fully implemented and tested.

Instead, the experiments performed are hard-coded in `quick_block_runner.py`, ran using argparse `-a`, `-b`, `-c`

### Custom Datasets

`dataset_selection.ipynb` is a jupyter notebook meant for creating the raw datasets. Replace the variables in the second cell for the desired dataset configuration.

`dataset_preprocessing.ipynb` is then used to preprocess the dataset. Likewise, the second notebook cell contains all the configurable parameters.

### Custom Experiments 

Replace ellipsis with desired configuration: 

```python
from modelling.experiment_wrapper import ExperimentConfig, ExperimentRunner
from raw_data.dataset import Dataset

# Load data
dataset = Dataset.load(...)
train, val, test = ...

# Configure experiment
config = ExperimentConfig(
    model_cls=...
    model_kwargs=...
)

# Run experiment
runner = ExperimentRunner(
    raw_train=train,
    raw_val=val,
    raw_test=test,
    experiment_config=config
)
runner.fit().test()
```

## Data

The project uses World Bank Development Indicators (WDI) data, but is extensible to use of other sources. The data processing pipeline includes:

- **Missing data imputation** (mean, linear interpolation, MICE, gain)
- **Scaling strategies** (z-score, robust scaling)

## Models

### Implemented Architectures

1. **LSTM/GRU Networks** - Recurrent neural networks
2. **Informer** - Transformer meant for long sequences
3. **SpaceTimeFormer** - Spatio-temporal transformer architecture

Additional architecture are relatively easy to plug in once implemented.

### Modelling Features

- Configurable window sizes and forecast horizons
- Support for multivariate time series forecasting
- Integrated preprocessing pipeline

## Core Components

### Data Pipeline (`raw_data/`)

Adapted from ISP, had no longer needed functionality that had been partially excised, explaining some of the design choices. 

**Main Classes:**
- `Core`: Controller class orchestrating the downloading and persistent storage
- `Dataset`: Dataset container/wrapper for economic time series

**Key Features:**
- Automatic data downloading from World Bank API
- Flexible country and indicator selection
- Time series alignment and missing data handling
- HDF5 storage for efficient data access

### Models (`modelling/`)

**Implemented Architectures:**

1. **RNN Models** (`models/rnn.py`)
   - LSTM and GRU variants
   - Configurable hidden dimensions and layers
   - Dropout and regularization support

2. **Informer** (`models/informer.py`)
   - Attention-based transformer
   - Efficient long sequence processing
   - Distilling mechanism for improved efficiency

3. **SpaceTimeFormer** (`models/spacetimeformer.py`)
   - Spatio-temporal attention
   - Global and local attention patterns
   - Designed for multivariate forecasting

**Training Infrastructure:**
- `ForecastModule`: PyTorch Lightning wrapper
- `TimeSeriesDataModule`: Data loading and batching
- `ExperimentRunner`: Complete training pipeline
- `ExperimentJob`: Single experiment execution
- `ExperimentSuite`: Batch experiment management

The later two, due to time constraints weren't extensively tested, originally intended to be used with scheduling datacenter computing jobs. Instead, the thesis's experiments used `ExperimentRunner` directly.

### Preprocessing (`preprocessing/`)

**Imputation Methods:**
- **Mean Imputation**: Simple mean filling
- **Linear Interpolation**: Time-aware interpolation
- **MICE**: Multiple Imputation by Chained Equations
- **GAIN**: Generative Adversarial Imputation Networks

**Scaling Methods:**
- **Z-Score**: Standard normalization
- **Robust**: Median and IQR-based scaling
- **Min-Max**: Range normalization

