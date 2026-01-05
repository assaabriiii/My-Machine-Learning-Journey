import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, in_features, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.linear(x)
