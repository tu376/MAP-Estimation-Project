import os
import csv
import random
from typing import Any, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


ArrayLike = Union[np.ndarray, Sequence[float], Sequence[int]]


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible numeric experiments."""
    random.seed(seed)
    np.random.seed(seed)


def to_numpy(data: ArrayLike, dtype: Optional[type] = None) -> np.ndarray:
    """Convert a sequence-like object to a NumPy array."""
    array = np.array(data)
    return array.astype(dtype) if dtype is not None else array


def clip_probability(theta: ArrayLike, eps: float = 1e-10) -> np.ndarray:
    """Clip probability values to the open interval (eps, 1-eps)."""
    theta_arr = np.array(theta, dtype=float)
    return np.clip(theta_arr, eps, 1.0 - eps)


def load_csv(path: str, usecols: Optional[Iterable[str]] = None, index_col: Optional[Union[int, str]] = None) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV file not found: {path}")
    return pd.read_csv(path, usecols=usecols, index_col=index_col)


def add_bias_column(X: ArrayLike, prepend: bool = True) -> np.ndarray:
    """Add a constant bias column of ones to the design matrix."""
    X_arr = np.asarray(X, dtype=float)
    ones = np.ones((X_arr.shape[0], 1), dtype=float)
    return np.concatenate([ones, X_arr], axis=1) if prepend else np.concatenate([X_arr, ones], axis=1)


def standardize(X: ArrayLike, mean: Optional[np.ndarray] = None, std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features to zero mean and unit variance."""
    X_arr = np.asarray(X, dtype=float)
    mean = np.mean(X_arr, axis=0) if mean is None else np.asarray(mean, dtype=float)
    std = np.std(X_arr, axis=0, ddof=0) if std is None else np.asarray(std, dtype=float)
    std_safe = np.where(std == 0, 1.0, std)
    return (X_arr - mean) / std_safe, mean, std_safe


def normalize(X: ArrayLike, min_val: Optional[np.ndarray] = None, max_val: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Normalize features to the range [0, 1]."""
    X_arr = np.asarray(X, dtype=float)
    min_val = np.min(X_arr, axis=0) if min_val is None else np.asarray(min_val, dtype=float)
    max_val = np.max(X_arr, axis=0) if max_val is None else np.asarray(max_val, dtype=float)
    range_val = np.where(max_val - min_val == 0, 1.0, max_val - min_val)
    return (X_arr - min_val) / range_val, min_val, max_val


def train_test_split(
    X: ArrayLike,
    y: ArrayLike,
    test_size: float = 0.2,
    random_state: Optional[int] = None,
    shuffle: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train and test sets."""
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError("X and y must have the same number of samples.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be a float between 0 and 1.")

    n_samples = X_arr.shape[0]
    n_test = int(np.ceil(n_samples * test_size))
    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    return X_arr[train_idx], X_arr[test_idx], y_arr[train_idx], y_arr[test_idx]


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute mean squared error between true and predicted values."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true_arr - y_pred_arr) ** 2))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Compute coefficient of determination (R²)."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    ss_res = np.sum((y_true_arr - y_pred_arr) ** 2)
    ss_tot = np.sum((y_true_arr - np.mean(y_true_arr)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def bernoulli_mle(data: ArrayLike) -> float:
    """Closed-form maximum likelihood estimator for Bernoulli probability."""
    data_arr = np.asarray(data, dtype=float)
    return float(np.mean(data_arr))


def beta_bernoulli_map(data: ArrayLike, alpha: float = 2.0, beta: float = 2.0) -> float:
    """Closed-form MAP estimator for Bernoulli-Beta model."""
    data_arr = np.asarray(data, dtype=float)
    k = np.sum(data_arr)
    n = data_arr.shape[0]
    if n + alpha + beta - 2 == 0:
        raise ValueError("Invalid prior parameters or empty data for beta-bernoulli MAP.")
    return float((k + alpha - 1.0) / (n + alpha + beta - 2.0))


def batch_generator(
    X: ArrayLike,
    y: Optional[ArrayLike] = None,
    batch_size: int = 32,
    shuffle: bool = True,
) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray]]]:
    """Yield minibatches from X and optional y."""
    X_arr = np.asarray(X)
    n_samples = X_arr.shape[0]
    indices = np.arange(n_samples)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        batch_idx = indices[start_idx : start_idx + batch_size]
        if y is None:
            yield X_arr[batch_idx], None
        else:
            y_arr = np.asarray(y)
            yield X_arr[batch_idx], y_arr[batch_idx]


__all__ = [
    "set_seed",
    "to_numpy",
    "clip_probability",
    "load_csv",
    "add_bias_column",
    "standardize",
    "normalize",
    "train_test_split",
    "mean_squared_error",
    "r2_score",
    "bernoulli_mle",
    "beta_bernoulli_map",
    "batch_generator",
]
