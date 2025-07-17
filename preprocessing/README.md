# Preprocessing

This directory contains the data preprocessing pipeline for time series economic data.

```
preprocessing/
├── preprocess.py                   # Main preprocessing classes
├── gain.py                         # GAIN imputation implementation
├── ...                             # Different approaches to implementing GAIN
```
## Core Components

`preprocess.py` is the module used by the rest of the codebase for preprocessing, containing separate `Scaler`, `Imputer` classes that are managed by the `Preprocessor`. They encapsulate a variety of methods in a unified interface for quick experimentation.

The various `gain` files were attempts at making an implementation of GAIN that would be suitable for the dataset used in the thesis, in particular large feature - low sample size.