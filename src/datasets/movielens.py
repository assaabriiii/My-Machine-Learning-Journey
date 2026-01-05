import pandas as pd
import numpy as np

def load_movielens_100k(path="data/ml-100k/u.data"):
    df = pd.read_csv(path, sep="\t", names=["user", "item", "rating", "timestamp"])
    df["user"] -= 1
    df["item"] -= 1
    return df


def make_implicit(df, threshold=4):
    df = df.copy()
    df["label"] = (df["rating"] >= threshold).astype(int)
    return df[df["label"] == 1][["user", "item", "timestamp"]]
