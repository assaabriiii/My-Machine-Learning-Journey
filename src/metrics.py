import numpy as np

def recall_at_k(rankings, ground_truth, k=10):
    hits = 0
    for user in ground_truth:
        if ground_truth[user] in rankings[user][:k]:
            hits += 1
    return hits / len(ground_truth)


def ndcg_at_k(rankings, ground_truth, k=10):
    total = 0
    for user in ground_truth:
        recs = rankings[user][:k]
        gt = ground_truth[user]
        if gt in recs:
            rank = recs.index(gt)
            total += 1 / np.log2(rank + 2)
    return total / len(ground_truth)
