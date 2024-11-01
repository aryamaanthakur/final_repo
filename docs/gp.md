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


## Configuration

YAML file could also be used for setting configuration.
Example: `config = GpGomeaConfig.from_yaml(file_path)`

| Parameter                | Default Value                              |
|--------------------------|--------------------------------------------|
| **EplexConfig**          |                                            |
| `selection`              | `'epsilon_lexicase'`                       |
| `lex_eps_global`         | `False`                                    |
| `lex_eps_dynamic`        | `False`                                    |
| `islands`                | `False`                                    |
| `num_islands`            | `10`                                       |
| `island_gens`            | `100`                                      |
| `verbosity`              | `0`                                        |
| `print_data`             | `False`                                    |
| `elitism`                | `True`                                     |
| `pHC_on`                 | `True`                                     |
| `prto_arch_on`           | `True`                                     |
| `max_len`                | `64`                                       |
| `max_len_init`           | `20`                                       |
| `popsize`                | `500`                                      |
| `g`                      | `500`                                      |
| `time_limit`             | `120` (2 minutes)                          |
| **FeatConfig**           |                                            |
| `pop_size`               | `100`                                      |
| `gens`                   | `100`                                      |
| `max_time`               | `120` (2 minutes)                          |
| `max_depth`              | `6`                                        |
| `verbosity`              | `2`                                        |
| `batch_size`             | `100`                                      |
| `functions`              | `['+', '-', '*', '/', '^2', '^3', 'sqrt', 'sin', 'cos', 'exp', 'log']` |
| `otype`                  | `'f'`                                      |
| **GpGomeaConfig**        |                                            |
| `t`                      | `3600` (1 hour)                            |
| `g`                      | `-1` (unlimited)                           |
| `e`                      | `499500`                                   |
| `finetune_max_evals`     | `500`                                      |
| `finetune`               | `True`                                     |
| `tour`                   | `4`                                        |
| `d`                      | `4`                                        |
| `pop`                    | `1024`                                     |
| `disable_ims`            | `True`                                     |
| `feat_sel`               | `20`                                       |
| `no_univ_exc_leaves_fos` | `False`                                    |
| `no_large_fos`           | `True`                                     |
| `bs`                     | `100`                                      |
| `fset`                   | `'+,-,*,/,log,sqrt,sin,cos'`               |
| `cmp`                    | `0.0`                                      |
| `rci`                    | `0.0`                                      |
| `verbose`                | `True`                                     |
| `random_state`           | `0`                                        |
| **GpLearnConfig**        |                                            |
| `population_size`        | `5000`                                     |
| `generations`            | `20`                                       |
| `stopping_criteria`      | `0.01`                                     |
| `p_crossover`            | `0.7`                                      |
| `p_subtree_mutation`     | `0.1`                                      |
| `p_hoist_mutation`       | `0.1`                                      |
| `p_point_mutation`       | `0.05`                                     |
| `max_samples`            | `0.9`                                      |
| `verbose`                | `1`                                        |
| `parsimony_coefficient`  | `0.01`                                     |
| `function_set`           | `['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'neg', 'inv', 'sin', 'cos', 'tan']` |
| `random_state`           | `42`                                       |
