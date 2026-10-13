"""W05. Naive Bayes appropriate to your representation. Implement AT LEAST ONE of the two
classes below and leave the other raising NotImplementedError.

Interface (Appendix A):
    class GaussianNB(var_smoothing=1e-9)       continuous features; per-class variances use ddof = 0
                                               plus var_smoothing * (largest feature variance of X)
    class MultinomialNB(alpha=1.0)             count features, Laplace/Lidstone smoothing
        fit(X, y) -> self; attribute classes_ (sorted)
        predict_log_proba(X) -> (n, K) normalised log-posteriors, columns in classes_ order
        predict(X) -> labels from classes_
"""
import numpy as np


def normal_log(log_values):
    max_log = np.max(log_values, axis=1, keepdims=True)
    return log_values - max_log - np.log(
        np.sum(np.exp(log_values - max_log), axis=1, keepdims=True)
    )


class GaussianNB:
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.theta_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        self.sigma_ = np.array([X[y == c].var(axis=0, ddof=0) for c in self.classes_])
        self.priors_ = np.array([np.mean(y == c) for c in self.classes_])
        self.epsilon_ = self.var_smoothing * np.max(X.var(axis=0, ddof=0))
        return self
        raise NotImplementedError

    def predict_log_proba(self, X):
        probabilities = []
        for i, c in enumerate(self.classes_):
            mean = self.theta_[i]
            var = self.sigma_[i] + self.epsilon_
            log_prior = np.log(self.priors_[i])
            log_likelihood = -0.5 * np.sum(np.log(2 * np.pi * var)) - 0.5 * np.sum(((X - mean) ** 2) / var, axis=1)
            probabilities.append(log_prior + log_likelihood)
        return normal_log(np.array(probabilities).T)
        raise NotImplementedError

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
        raise NotImplementedError


class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.class_count_ = np.array([np.sum(y == c) for c in self.classes_])
        self.feature_count_ = np.array([np.sum(X[y == c], axis=0) for c in self.classes_])
        feature_totals = self.feature_count_.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(
            (self.feature_count_ + self.alpha)
            / (feature_totals + self.alpha * X.shape[1])
        )
        self.class_log_prior_ = np.log(self.class_count_ / np.sum(self.class_count_))
        return self
        raise NotImplementedError

    def predict_log_proba(self, X):
        joint_log_proba = X @ self.feature_log_prob_.T + self.class_log_prior_
        return normal_log(joint_log_proba)
        raise NotImplementedError

    def predict(self, X):
        log_proba = self.predict_log_proba(X)
        return self.classes_[np.argmax(log_proba, axis=1)]
        raise NotImplementedError
