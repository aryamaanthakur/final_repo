# Data Preparation

This document outlines how to prepare data for training the models, detailing both command-line usage and Jupyter notebook examples. This documentation expects the data to be in the same format as AIFeynman dataset.

## Data Preparation Options

### Using the command line tool
There are two ways to use the command line tool for preparing data. Either supply each cli-argument individually or use a YAML configuration file.

Create a `config.yaml` file resembling this snippet.

```yaml
output_dir: data
encoder_vocab: ./algorithms/xval-transformers/encoder_vocab
decoder_vocab: ./algorithms/xval-transformers/decoder_vocab
xval: true
chunk_size: 400
max_len: 11
df_path: ./FeynmanEquationsModified.csv
```


Run the following command for data preparation.

$ `python prepare_dataset.py --config_file <path_to_config_file>`

Or pass each argument like this

$ `python prepare_dataset.py --output_dir data --chunk_size 400 --xval --df_path FeynmanEquationsModified.csv`


## Configuration

| Parameter       | Description                                   | Default                                 |
|-----------------|-----------------------------------------------|-----------------------------------------|
| `max_len`       | Maximum length of input sequences, used for padding in vanilla-transformers             | `33`                                    |
| `xval`          | Enable xVal Encoding or not                  | `False`                                  |
| `chunk_size`    | Size of data chunks                          | `400`                                   |
| `df_path`       | Path to the main data file                   | `"./FeynmanEquationsModified.csv"`      |
| `output_dir`    | Directory to save output files               | `"./data"`                              |
| `encoder_vocab` | Path to the encoder vocabulary file          | `"./algorithms/transformer/encoder_vocab"`                     |
| `decoder_vocab` | Path to the decoder vocabulary file          | `"./algorithms/transformer/decoder_vocab"` |
