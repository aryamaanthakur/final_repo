# Hybrid Predictor

This document outlines the usage of HybridPredictor.

## Usage

You can also refer to `hybrid-predictor.ipynb` in examples directory.

Step 1: Create configuration
```python
from algorithms.xval_transformers import Config
config = Config(experiment_name="test", ...) # YAML file could also be used
```

Step 2: Create Predictor
```python
from algorithms.hybrid import HybridPredictor
predictor = HybridPredictor(config)
```

Step 3: Predict
```python
eqn, r2 = predictor.predict_equation(x, y)
```

## Configuration
Predictor uses the same configuration as Trainer because it creates the exact same model and loads the weights from provided location.

However, following are some predictor specific parameters

| Parameter               | Description                                       | Default                     |
|-------------------------|---------------------------------------------------|-----------------------------|
| **Hybrid Parameters**   |                                                   |                             |
| `pop_size`              | Population size for genetic programming           | `500`                       |
| `cxpb`                  | Crossover probability                             | `0.7`                       |
| `mutpb`                 | Mutation probability                              | `0.2`                       |
| `num_generations`       | Number of generations for genetic programming     | `15`                        |
| `gp_verbose`            | Verbosity of genetic programming                  | `False`                     |
| `beam_size`             | Beam size for beam search                         | `5`                         |
| `num_equations`         | Number of equations to generate                   | `20`                        |