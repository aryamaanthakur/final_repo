# Training

This document outlines how to train models in the library, detailing both command-line usage and Jupyter notebook examples.

## Training Options

### Using the command line tool
There are two ways to use the command line tool for training transformers. Either supply each cli-argument individually or use a YAML configuration file.

Create a `config.yaml` file resembling this snippet.

```yaml
# Experiment details
experiment_name: "test"
root_dir: "/pscratch/sd/a/aryamaan"
device: "cuda:0"

# Training parameters
epochs: 1
seed: 42
use_half_precision: true

# Data parameters
train_batch_size: 256
test_batch_size: 256
train_split: 0.8
test_split: 0.1
primary_df: ./FeynmanEquationsModified.csv
train_df: ./data/train_df.csv
data_dir: ./data
chunk_size: 400

# Scheduler parameters
scheduler_type: "cosine_annealing" 
T_0: 10
T_mult: 1
T_max: 125000

# Optimizer parameters
optimizer_type: "adam" 
optimizer_lr: 5.0e-5
optimizer_momentum: 0.9
optimizer_weight_decay: 0.0001
clip_grad_norm: -1

# Model parameters
model_name: "seq2seq_transformer"
xval: true
embedding_size: 64
hidden_dim: 64
nhead: 8
num_encoder_layers: 2
num_decoder_layers: 6
dropout: 0.2
input_emb_size: 64
max_input_points: 11
src_vocab_size: 3
tgt_vocab_size: 59

# Criterion
criterion: "cross_entropy"

# Hybrid parameters
pop_size: 500
cxpb: 0.7
mutpb: 0.2
num_generations: 15
gp_verbose: true
beam_size: 5
num_equations: 20
```


Run the following command for training.

$ `python main.py --config_file <path_to_config_file>`

Or pass each argument like this

$ `python main.py --experiment_name <experiment_name> --root_dir <root_dir> ...`

### Using Jupyter Notebooks
If you want to use trainer in a jupyter notebook or utilize it somewhere differently, following the 3 step process :-\


Step 1: Create a configuration
```python
from algorithms.xval_transformers import Config
config = Config(experiment_name="test", ...) # YAML file could also be used
```

Step 2: Create dataset and dataloader
```python
datasets, train_equations, test_equations = get_datasets(...)
dataloaders = get_dataloaders(...)
```

Step 3: Create Trainer
```python
trainer = Trainer(config, dataloaders)
trainer.train()
```

Now you are ready to train a transformer. Refer to `train-xval-transformers.ipynb` or `train-vanilla-transformers.ipynb` for detailed usage.

## Configuration

| Parameter               | Description                                       | Default                     |
|-------------------------|---------------------------------------------------|-----------------------------|
| `experiment_name`       | Name of the experiment                            | `"test"`                    |
| `root_dir`              | Root directory for the project                    | `"./"`                      |
| `device`                | Device to use for training                        | `"cuda:0"`                  |
| **Data Parameters**     |                                                   |                             |
| `train_batch_size`      | Batch size for training                           | `256`                       |
| `test_batch_size`       | Batch size for testing                            | `256`                       |
| `train_split`           | Proportion of data for training                   | `0.8`                       |
| `test_split`            | Proportion of data for testing (totally unseen)   | `0.1`                       |
| `primary_df`            | Path to the primary dataset                       | `"./FeynmanEquationsModified.csv"` |
| `train_df`              | Path to the training dataset                      | `"./data_400/train_df.csv"` |
| `data_dir`              | Directory for data files                          | `"./data_400"`              |
| `chunk_size`            | Chunk size for data                               | `400`                       |
| **Training Parameters** |                                                   |                             |
| `epochs`                | Number of training epochs                         | `10`                        |
| `seed`                  | Random seed for reproducibility                   | `42`                        |
| `use_half_precision`    | Whether to use half-precision training            | `True`                      |
| **Scheduler Parameters**|                                                   |                             |
| `scheduler_type`        | Type of learning rate scheduler. Choose between `multi_step`, `reduce_lr_on_plateau`, `cosine_annealing_warm_restart`, `cosine_annealing` and `none` | `"cosine_annealing"`        |
| `T_0`                   | Initial step for the cosine annealing with warm restarts | `10`                        |
| `T_mult`                | Multiplicative factor for cosine annealing with warm restarts | `1`                         |
| `T_max`                 | Maximum steps for the cosine annealing scheduler  | `18750`                     |
| **Optimizer Parameters**|                                                   |                             |
| `optimizer_type`        | Type of optimizer. Choose between `adam`, `adamw` and `sgd` | `"adam"`                    |
| `optimizer_lr`          | Learning rate for the optimizer                   | `5e-5`                      |
| `optimizer_momentum`    | Momentum for SGD optimizer                        | `0.9`                       |
| `optimizer_weight_decay`| Weight decay for the optimizer                    | `0.0001`                    |
| `clip_grad_norm`        | Max norm for gradient clipping                    | `-1`                        |
| **Model Parameters**    |                                                   |                             |
| `model_name`            | Name of the model architecture                    | `"seq2seq_transformer"`     |
| `xval`                  | Whether to use xVal Encoding for numbers or not   | `True`                      |
| `embedding_size`        | Size of embeddings                                | `64`                        |
| `hidden_dim`            | Hidden layer dimension                            | `64`                        |
| `nhead`                 | Number of attention heads                         | `8`                         |
| `num_encoder_layers`    | Number of encoder layers                          | `2`                         |
| `num_decoder_layers`    | Number of decoder layers                          | `6`                         |
| `dropout`               | Dropout rate                                      | `0.2`                       |
| `input_emb_size`        | Input embedding size                              | `64`                        |
| `max_input_points`      | Maximum number of input points                    | `11`                        |
| `src_vocab_size`        | Source vocabulary size                            | `3`                         |
| `tgt_vocab_size`        | Target vocabulary size                            | `59`                        |
| **Criterion**           |                                                   |                             |
| `criterion`             | Loss criterion                                    | `"cross_entropy"`           |
| **Hybrid Parameters**   |                                                   |                             |
| `pop_size`              | Population size for genetic programming           | `500`                       |
| `cxpb`                  | Crossover probability                             | `0.7`                       |
| `mutpb`                 | Mutation probability                              | `0.2`                       |
| `num_generations`       | Number of generations for genetic programming     | `15`                        |
| `gp_verbose`            | Verbosity of genetic programming                  | `False`                     |
| `beam_size`             | Beam size for beam search                         | `5`                         |
| `num_equations`         | Number of equations to generate                   | `20`                        |

Note: Some parameters like Hybrid Parameters are not used while training.