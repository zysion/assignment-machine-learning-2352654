"""W03 (release catch-up). One impurity/split routine for decision trees.

Interface (Appendix A):
    entropy(y) -> float                 Shannon entropy in bits of the label array y.
    gini(y) -> float                    Gini impurity of y.
    information_gain(y, mask) -> float  Entropy of y minus the weighted entropy of y[mask], y[~mask].
    best_threshold(x, y) -> (t, gain)   Best split x <= t on one continuous feature by information
                                        gain; t is a midpoint between two consecutive distinct values.
                                        Return (None, 0.0) when x has a single distinct value.
"""
import numpy as np


def entropy(y):
    entropy = -sum((np.sum(y == i) / len(y)) * np.log2(np.sum(y == i) / len(y)) for i in np.unique(y))
    return entropy
    raise NotImplementedError


def gini(y):
    gini = 1 - sum((np.sum(y == i) / len(y)) ** 2 for i in np.unique(y))
    return gini
    raise NotImplementedError


def information_gain(y, mask):
    entropy_y = entropy(y)
    entropy_masked = entropy(y[mask])
    entropy_unmasked = entropy(y[~mask])
    weighted_entropy = (len(y[mask]) / len(y)) * entropy_masked + (len(y[~mask]) / len(y)) * entropy_unmasked
    return entropy_y - weighted_entropy
    raise NotImplementedError


def best_threshold(x, y):
    unique_values = np.unique(x)
    if len(unique_values) == 1:
        return None, 0.0
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2
    best_gain = -np.inf
    best_threshold = None
    for t in thresholds:
        mask = x <= t
        gain = information_gain(y, mask)
        if gain > best_gain:
            best_gain = gain
            best_threshold = t
    return best_threshold, best_gain
    raise NotImplementedError
