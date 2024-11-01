from dataclasses import dataclass, field, fields
from typing import Optional, List

@dataclass
class Config:
    experiment_name: Optional[str] = "test"
    root_dir: Optional[str] = "./"
    device: Optional[str] = "cuda:0"
    
    #data parameters
    train_batch_size: Optional[int] = 256
    test_batch_size:Optional[int] = 256
    train_split: Optional[float] = 0.8
    test_split: Optional[float] = 0.1
    primary_df: Optional[str] = "./FeynmanEquationsModified.csv"
    train_df: Optional[str] = "./data_400/train_df.csv"
    data_dir: Optional[str] = "./data_400"
    chunk_size: Optional[int] = 400

    #training parameters
    epochs: Optional[int] = 10
    seed: Optional[int] = 42
    use_half_precision: Optional[bool] = True

    # scheduler parameters
    scheduler_type: Optional[str] = "cosine_annealing" # multi_step or none
    T_0: Optional[int] = 10
    T_mult: Optional[int] = 1
    T_max: Optional[int] = 18750

    # optimizer parameters
    optimizer_type: Optional[str] = "adam" # sgd or adam
    optimizer_lr: Optional[float] = 5e-5   
    optimizer_momentum: Optional[float] = 0.9
    optimizer_weight_decay: Optional[float] = 0.0001
    clip_grad_norm: Optional[float] = -1
        
    # Model Parameters
    model_name: Optional[str] = "seq2seq_transformer"
    xval: Optional[bool] = True
    embedding_size: Optional[int] = 64
    hidden_dim: Optional[int] = 64
    nhead: Optional[int] = 8
    num_encoder_layers: Optional[int] = 2
    num_decoder_layers: Optional[int] = 6
    dropout: Optional[int] = 0.2
    input_emb_size: Optional[int] = 64
    max_input_points: Optional[int] = 11
    src_vocab_size: Optional[int] = 3
    tgt_vocab_size: Optional[int] = 59

    # Criterion
    criterion: Optional[str] = "cross_entropy"
    
    # Hybrid
    pop_size: Optional[int] = 500
    cxpb: Optional[float] = 0.7
    mutpb: Optional[float] = 0.2 
    num_generations: Optional[int] = 15
    gp_verbose: Optional[bool] = False
    beam_size: Optional[int] = 5
    num_equations: Optional[int] = 20


    def print_config(self):
        print("="*50+"\nConfig\n"+"="*50)
        for field in fields(self):
            print(field.name.ljust(30), getattr(self, field.name))
        print("="*50)

    def save(self, root_dir):
        path = root_dir + "/config.txt"
        with open(path, "w") as f:
            f.write("="*50+"\nConfig\n"+"="*50 + "\n")
            for field in fields(self):
                f.write(field.name.ljust(30) + ": " + str(getattr(self, field.name)) + "\n")
            f.write("="*50)   
