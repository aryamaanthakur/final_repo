import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils import EncoderTokenizer, DecoderTokenizer

PAD_IDX = 0

def prepare_dataset(config):
    """
    Prepare the dataset for training.

    Args:
    - config: Configuration object containing dataset parameters

    Returns:
    - train_df: DataFrame containing training data information
    - equations_df: DataFrame containing equations information
    """
    input_max_len = config.chunk_size
    df = pd.read_csv(config.df_path)

    encoder_tokenizer = EncoderTokenizer(config.encoder_vocab, config.max_len)
    decoder_tokenizer = DecoderTokenizer(config.decoder_vocab)

    train_df = {
        "filename":[],
        "data_num":[], 
        "number":[]
        }
    
    for (index, row) in tqdm(df.iterrows()):
        # Read data from file
        with open(row["path"]) as file:
            data = file.readlines()
        
        # Tokenize the data using encoder tokenizer
        X = encoder_tokenizer.tokenize(data)

        n_splits = X.shape[0] // input_max_len
        X = X[:n_splits*input_max_len]
        x_chunks = np.split(X, n_splits)

        sub_dir = os.path.join(config.output_dir, row["Filename"])
        os.makedirs(sub_dir, exist_ok=True)

        # Save tokenized data chunks to files
        for (index, x) in enumerate(x_chunks):
            np.save(os.path.join(sub_dir, f"{index}.npy"), x)

        # Update train_df with data information
        train_df["filename"].extend([row["Filename"]]*n_splits)
        train_df["data_num"].extend([i for i in range(n_splits)])
        train_df["number"].extend([row["Number"] for i in range(n_splits)])

    train_df = pd.DataFrame(train_df)

    equations_df = {
        "filename":[],
        "prefix":[],
        "encoded":[]
        }
    
    prefix_equations = np.zeros((100, 256)).astype(np.int32)
    for (index, row) in df.iterrows():
        equations_df["filename"].append(row["Filename"])
        prefix = eval(row["prefix"])
        prefix = ["<bos>"] + prefix + ["<eos>"]
        equations_df["prefix"].append(prefix)
        print(row["Filename"])
        y = decoder_tokenizer.encode([prefix])[0]
        y = np.pad(y, (0, 256 - len(y)))
        prefix_equations[row["Number"]-1, :] = y
        equations_df["encoded"].append(y)

    path = os.path.join(config.output_dir, "prefix_equations.npy")
    np.save(path, prefix_equations)
    equations_df = pd.DataFrame(equations_df)

    return train_df, equations_df

class FeynmanDataset(Dataset):
    """
    Custom Dataset class for Feynman Equation dataset.
    """
    def __init__(self, df, dataset_dir):
        super().__init__()
        self.df = df
        self.dataset_dir = dataset_dir
        self.prefix_equations = np.load(os.path.join(dataset_dir, "prefix_equations.npy"))

        prefix_equations = []
        for prefix in self.prefix_equations:
            prefix_equations.append(np.trim_zeros(prefix))

        self.prefix_equations = prefix_equations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Get a single item from the dataset.

        Args:
        - idx: Index of the item

        Returns:
        - Tuple containing input and target data
        """
        row = self.df.iloc[idx]
        path = os.path.join(os.path.join(self.dataset_dir, row['filename']), f"{row['data_num']}.npy")
        x = np.load(path).astype(np.int32)

        path = os.path.join(self.dataset_dir, f"{row['filename']}.npy")
        y = self.prefix_equations[row['number'] - 1]

        return (torch.Tensor(x).long(), torch.Tensor(y).long())

def get_datasets(df, input_df, dataset_dir, split):
    """
    Prepare training, validation, and test datasets.

    Args:
    - df: DataFrame containing data information
    - input_df: DataFrame containing input data information
    - dataset_dir: Directory containing dataset files
    - split: List containing train, validation, and test split ratios

    Returns:
    - datasets: Dictionary containing train, validation, and test datasets
    - train_equations: List of equations used for training
    - test_equations: List of equations used for testing
    """
    assert sum(split) == 1.0, "split ratios should sum to 1"
    
    train_df, test_df = train_test_split(df, test_size=split[2], shuffle=True, random_state=42)
    train_equations = train_df['Filename'].tolist()
    test_equations = test_df['Filename'].tolist()
    
    input_test_df = input_df[input_df['filename'].isin(test_equations)]
    input_train_df = input_df[input_df['filename'].isin(train_equations)]

    val_size = split[1]/(split[0] + split[1])
    input_train_df, input_val_df = train_test_split(input_train_df, test_size=val_size, shuffle=True, random_state=42)

    train_dataset = FeynmanDataset(input_train_df, dataset_dir)
    val_dataset = FeynmanDataset(input_val_df, dataset_dir)
    test_dataset = FeynmanDataset(input_test_df, dataset_dir)

    datasets = {
        "train":train_dataset,
        "test":test_dataset,
        "valid":val_dataset
        }
    
    return datasets, train_equations, test_equations

def get_dataloaders(datasets, train_bs, val_bs, test_bs):
    """
    Get data loaders for training, validation, and testing.

    Args:
    - datasets: Dictionary containing train, validation, and test datasets
    - train_bs: Batch size for training
    - val_bs: Batch size for validation
    - test_bs: Batch size for testing

    Returns:
    - dataloaders: Dictionary containing train, validation, and test data loaders
    """
    train_dataloader = DataLoader(datasets['train'], batch_size=train_bs,
                                  shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_dataloader = DataLoader(datasets['valid'], batch_size=val_bs,
                                  shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(datasets['test'], batch_size=test_bs,
                                  shuffle=False, num_workers=2, pin_memory=False, collate_fn=collate_fn)
    
    dataloaders = {
        "train":train_dataloader,
        "test":test_dataloader,
        "valid":val_dataloader
        }
    
    return dataloaders

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for (src_sample, tgt_sample) in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
        
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    return src_batch, tgt_batch
