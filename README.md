# Evolutionary and Transformer Models for Symbolic Regression

## Overview
This project is a ready-to-use tool for Symbolic Regression. The main feature of this library is the ability to train a transformer model for predicting equations given datapoints. Moreover, this library also includes a hybrid predictor which utilizes genetic programming to enhance the output of transformers.

## Features
- Train vanilla transformers
- Train transformers with xVal encoding
- High level API wrappers for 4 GP methods
- Hybrid Predictor

## Installation

Create a virtual environment with `python3.10+` and install required libraries.

$ ```pip install -r requirements.txt```

Note: For installation of required packages for specific GP methods, refer to documentation.

## Quick Start

### Data Preparation
$ `python prepare_data.py --config_file ./dataconfig.yaml`

Or pass the arguments directly without config_file

$ `python prepare_data.py --output_dir data --encoder_vocab algorithms/xval-transformers/encoder_vocab --decoder_vocab algorithms/transformers/decoder_vocab --xval --chunk_size 400 --df_path FeynmanEquationsModified.csv`

### Train
$ `python main.py --config_file ./config.yaml`

Or pass the arguments directly without config_file. Refer documentation for all arguments.

## Pretrained Weights
Vanilla Transformer: link \
xVal Transformer: link

## Documentation
Full documentation can be found in the `docs` folder.