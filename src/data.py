"""Data loading and the frozen split. Fit every scaler, encoder or projection on the training part only."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def read_data(seed=31):
    # 1. read data
    data_dir = Path(__file__).resolve().parent.parent / "data" / "UCI HAR Dataset" / "UCI HAR Dataset" / "train"
    X = pd.read_csv(data_dir / "X_train.txt", sep=r"\s+", header=None)
    y = pd.read_csv(data_dir / "y_train.txt", sep=r"\s+", header=None)
    subjects = pd.read_csv(data_dir / "subject_train.txt", sep=r"\s+", header=None).squeeze()

    # 2. create GroupShuffleSplit to devide 80/20 follow subject ID with seed = 31
    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=seed)

    train_idx, val_idx = next(gss.split(X, y, groups=subjects))

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    return X_train, X_val, y_train, y_val