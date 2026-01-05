import numpy as np

def leave_one_out_split(interactions):
    interactions = interactions.sort_values("timestamp")

    train = []
    test = []

    for user, group in interactions.groupby("user"):
        if len(group) < 2:
            continue
        test.append(group.iloc[-1])
        train.append(group.iloc[:-1])

    train = np.concatenate([g.values for g in train])
    test = np.array([t.values for t in test])

    return train, test
