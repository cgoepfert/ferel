import numpy as np

def calculate_classification_metrics(X, y, hyp, offset):
    (n, d) = X.shape
    slack = np.maximum(0, 1 - y * (np.dot(X, hyp) - offset))
    num_miss = np.count_nonzero(slack > 1)
    acc = 1 - num_miss / n
    num_intrusions_pos = np.count_nonzero(np.logical_and(y == 1, slack > 0))
    num_intrusions_neg = np.count_nonzero(np.logical_and(y == -1, slack > 0))
    return slack, acc, num_miss, num_intrusions_pos, num_intrusions_neg
