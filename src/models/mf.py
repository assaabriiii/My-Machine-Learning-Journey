import torch
import torch.nn as nn

class MatrixFactorization(nn.Module):
    def __init__(self, n_users, n_items, dim=64):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, dim)
        self.item_emb = nn.Embedding(n_items, dim)

    def forward(self, users, items):
        u = self.user_emb(users)
        v = self.item_emb(items)
        return (u * v).sum(dim=1)
