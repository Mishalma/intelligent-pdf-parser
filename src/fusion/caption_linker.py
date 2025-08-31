from scipy.optimize import linear_sum_assignment
import numpy as np

def link_captions(captions, targets, window=100):  # targets = images + tables
    cost_matrix = np.inf * np.ones((len(captions), len(targets)))
    for i, cap in enumerate(captions):
        cap_y = cap['bbox'][3]  # Bottom of caption
        for j, tgt in enumerate(targets):
            tgt_y = tgt['bbox'][1]  # Top of target
            dist = abs(cap_y - tgt_y)
            if dist < window:
                cost_matrix[i, j] = dist
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    links = [(captions[i], targets[j]) for i, j in zip(row_ind, col_ind) if cost_matrix[i, j] < np.inf]
    return links