import argparse
import yaml
from dataclasses import dataclass, asdict
import os

@dataclass
class Config:
    max_len: int = 11
    xval: bool = True
    chunk_size: int = 400
    df_path: str = "./FeynmanEquationsModified.csv"
    output_dir: str = "./data"
    encoder_vocab: str = "./encoder_vocab"
    decoder_vocab: str = "./algorithms/xval_transformer/decoder_vocab"

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Argument parser for dataset preparation")

    # Config file argument
    parser.add_argument("--config_file", type=str, default=None, help="Path to YAML config file")
    
    # Other Config fields as arguments
    parser.add_argument("--chunk_size", type=int, default=400, help="Maximum length of input sequences")
    parser.add_argument("--max_len", type=int, default=11, help="Maximum length for processed data")
    parser.add_argument("--df_path", type=str, default="./FeynmanEquationsModified.csv", help="Path to the dataset CSV file")
    parser.add_argument("--output_dir", type=str, default="./data", help="Output directory for processed data")
    parser.add_argument("--encoder_vocab", type=str, default="./encoder_vocab", help="Path to encoder vocabulary file")
    parser.add_argument("--decoder_vocab", type=str, default="./decoder_vocab", help="Path to decoder vocabulary file")
    parser.add_argument("--xval", action="store_true", default=True, help="xVal")

    args = parser.parse_args()

    # Load config from YAML file if provided
    config_dict = vars(args)
    if args.config_file and os.path.isfile(args.config_file):
        with open(args.config_file, 'r') as file:
            file_config = yaml.safe_load(file)
            config_dict.update({k: v for k, v in file_config.items() if k in config_dict and v is not None})
    
    config_dict.pop("config_file")
    return Config(**config_dict)

if __name__ == "__main__":
    config = parse_args()

    if config.xval:
        from algorithms.xval_transformers.dataset import prepare_dataset
    else:
        from algorithms.transformers.dataset import prepare_dataset
    
    train_df, equations_df = prepare_dataset(config)

    train_df.to_csv(os.path.join(config.output_dir, "train_df.csv"))
    equations_df.to_csv(os.path.join(config.output_dir, "equations_df.csv"))
    
    print("Dataset preparation complete!")
