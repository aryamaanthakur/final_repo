# Genetic Programming Methods

This document outlines the usage of high level python wrappers over some genetic programming methods for symbolic regression. This library contains implementation of following methods :-
- gplearn
- GP-GOMEA
- Epsilon Lexicase Selection
- FEAT

## Usage

Here's a 3 step process for predicting equations with GP-GOMEA. Same process could be used for other 3 methods as well.

Step 1: Create Configuration
```python
from algorithms.gp.gpgomea import GpGomeaConfig
config = GpGomeaConfig(verbose=False, finetune=True)
```

Step 2: Create Regressor
```python
from algorithms.gp.gpgomea import GpGomeaRegressor
regressor = GpGomeaRegressor(config)
```

Step 3: Predict
```python
model = regressor.predict_single(X, y) # X and y are numpy arrays
```

## Installation

Some GP methods will require special installations and building for proper usage. Also, while testing I was only able to use certain libraries only on linux.

Refer to below table for the installation guides for specific methods and these were not included in requirements.txt


| Library            | Reference                          |
---------------------|-------------------------------------
|gplearn | https://gplearn.readthedocs.io/en/stable/installation.html                |
|GP GOMEA | https://github.com/marcovirgolin/gpg?tab=readme-ov-file#installation                |
|EPLEX | https://github.com/cavalab/ellyn?tab=readme-ov-file#quick-install |
|FEAT | https://github.com/cavalab/feat?tab=readme-ov-file#install-in-a-conda-environment                      |