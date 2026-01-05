import torch
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path

from src.config import load_config
from src.utils.seed import set_seed
from src.utils.device import get_device
from src.models.mf import MatrixFactorization


def bpr_loss(pos_scores, neg_scores):
    return -torch.log(torch.sigmoid(pos_scores - neg_scores)).mean()


def main(config_path):
    cfg = load_config(config_path)
    set_seed(cfg.experiment.seed)
    device = get_device()

    # ✅ Dummy sizes (you should compute from dataset)
    n_users = 943
    n_items = 1682

    model = MatrixFactorization(n_users, n_items, cfg.model.embedding_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    out_dir = Path("outputs/runs") / cfg.experiment.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ✅ Dummy data (replace with real train triples)
    users = torch.randint(0, n_users, (10000,))
    pos_items = torch.randint(0, n_items, (10000,))
    neg_items = torch.randint(0, n_items, (10000,))

    for epoch in range(cfg.train.epochs):
        model.train()

        total_loss = 0
        for i in tqdm(range(0, len(users), cfg.train.batch_size)):
            batch_users = users[i:i+cfg.train.batch_size].to(device)
            batch_pos = pos_items[i:i+cfg.train.batch_size].to(device)
            batch_neg = neg_items[i:i+cfg.train.batch_size].to(device)

            pos_scores = model(batch_users, batch_pos)
            neg_scores = model(batch_users, batch_neg)

            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{cfg.train.epochs} loss={total_loss:.4f}")

    torch.save(model.state_dict(), out_dir / "model.pt")
    print("✅ Saved model:", out_dir / "model.pt")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
