
from algorithms.xval_transformers.engine.config import Config
from algorithms.hybrid.hybrid import HybridPredictor
import pandas as pd
from tqdm import tqdm
import torch
from algorithms.xval_transformers.dataset import get_datasets, get_dataloaders


df = pd.read_csv("./FeynmanEquationsModified.csv")
input_df = pd.read_csv("data_400/train_df.csv")
datasets, train_equations, test_equations = get_datasets(df, input_df, "./data_400", [0.80, 0.1, 0.1])
dataloaders = get_dataloaders(datasets, 256, 256, 100)

config = Config(
        experiment_name="new_run7",
        device="cuda",
        use_half_precision=True,
        root_dir="/pscratch/sd/a/aryamaan",
        optimizer_weight_decay=0.0001,
        optimizer_lr=5e-5,
        T_max=782*2000,
        epochs=10,
        scheduler_type="cosine_annealing",
        T_0=int(6250*2.5),
        T_mult=1)

pbar = tqdm(dataloaders["test"], total=len(dataloaders["test"]))
pbar.set_description("Test")

predictor = HybridPredictor(config)

y_preds = []
y_true = []
count = 0
for src, num_array, tgt in pbar:
    count += 1
    if count <50:
        continue
    src = src.numpy()
    tgt = tgt.numpy()
    
    x = num_array[:4].reshape(-1, 11)
    column_of_ones = torch.where((x == 1).all(dim=0))[0][0]
    y = x[:, 0]
    x = x[:, 1:column_of_ones]
    print(f"{x.shape=} {y.shape=}")
    eqn = predictor.predict_equation(x.numpy(), y.numpy())
    print(eqn)

    y_preds.append(y_pred.cpu().numpy())
    y_true.append(np.trim_zeros(tgt[0]))
    print("pred", y_pred.cpu().tolist())
    print("true", y_true[-1].tolist())

    break

print(config)
