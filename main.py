import os
import yaml
import argparse
from typing import Optional, List
from algorithms.xval_transformers.engine.config import Config
from dataclasses import asdict
import pandas as pd

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Argument parser for model configuration")
    
    parser.add_argument("--config_file", type=str, default=None, help="Path to YAML config file")

    for field_name, field_value in Config.__dataclass_fields__.items():
        arg_type = field_value.type
        default_value = field_value.default
        help_text = f"{field_name} (default: {default_value})"
        
        if arg_type == Optional[bool]:  # for booleans, we add store_true or store_false
            parser.add_argument(f"--{field_name}", action="store_true", default=default_value, help=help_text)
        elif arg_type == Optional[str]:
            parser.add_argument(f"--{field_name}", type=str, default=default_value, help=help_text)
        else:
            parser.add_argument(f"--{field_name}", type=arg_type, default=default_value, help=help_text)
    
    args = parser.parse_args()
    config_dict = vars(args)

    if args.config_file and os.path.isfile(args.config_file):
        with open(args.config_file, 'r') as file:
            file_config = yaml.safe_load(file)
            config_dict.update({k: v for k, v in file_config.items() if k in config_dict and v is not None})

    config_dict.pop("config_file")
    return Config(**config_dict)

def main():
    config = parse_args()
    config.print_config()
    
    if config.xval:
        from algorithms.xval_transformers.dataset import get_datasets, get_dataloaders
        from algorithms.xval_transformers.engine import Trainer
    else:
        from algorithms.transformers.dataset import get_datasets, get_dataloaders
        from algorithms.transformers.engine import Trainer

    df = pd.read_csv(config.primary_df)
    input_df = pd.read_csv(config.train_df)

    datasets, train_equations, test_equations = get_datasets(
        df,
        input_df,
        config.data_dir,
        [config.train_split, 1 - config.train_split - config.test_split, config.test_split]
    )

    dataloaders = get_dataloaders(
        datasets,
        config.train_batch_size,
        config.train_batch_size,
        config.test_batch_size
    )

    trainer = Trainer(config, dataloaders)
    trainer.train()
    trainer.test_seq_acc()

if __name__ == "__main__":
    main()
