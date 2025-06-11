import numpy as np
from collections import Counter

def majority_vote(*pred_lists):
    preds_array = np.array(pred_lists).T  # shape (n_samples, n_models)

    final_preds = []
    for row in preds_array:
        vote = Counter(row).most_common(1)[0][0]
        final_preds.append(vote)
    return np.array(final_preds)
