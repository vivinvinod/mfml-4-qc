import numpy as np
import copy
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

from typing import Union, Dict, Callable


def _train_estimator(base_estimator: object, X: np.ndarray, y: np.ndarray) -> object:
    """Helper function to instantiate and train a model. Model must have train or fit attribute"""
    model = copy.deepcopy(base_estimator)
    if hasattr(model, "train"):
        model.train(X, y)
    elif hasattr(model, "fit"):
        model.fit(X, y)
    else:
        raise AttributeError(
            "Estimator must have a '.train(X, y)' or '.fit(X, y)' method."
        )
    return model


def _gpr_variance(model: object, X_train: np.ndarray, X_pool: np.ndarray) -> np.ndarray:
    """
    Computes exact predictive variance for the built-in KRR models using standalone kernels.

    V[f_*] = k(x_*, x_*) - k(X, x_*)^T (K(X, X) + reg * I)^{-1} k(X, x_*)

    """
    # training kernel
    K_train = model._generate_kernel(X_train)
    K_train[np.diag_indices_from(K_train)] += model.reg

    # pool kernel
    K_pool_train = model._generate_kernel(X_train=X_train, X_test=X_pool)

    # For standard stationary kernels (Matern, Gaussian, Laplacian),
    # the distance of a point to itself is 0, so exp(0) = 1.0. -> k(x_*, x_*)
    k_pool_pool_diag = 1.0

    # (K(X, X) + reg * I)^{-1} k(X, x_*)
    inv_K_train_dot_K_pt = np.linalg.solve(K_train, K_pool_train.T)

    # diagonal of the result
    variance = k_pool_pool_diag - np.sum(K_pool_train * inv_K_train_dot_K_pt.T, axis=1)

    return variance


def _uq_variance(model: object, X_pool: np.ndarray, X_train: np.ndarray) -> np.ndarray:
    """Helper function for Variance-based uncertainty quantification."""
    try:
        _, std_pool = model.predict(X_pool, return_std=True)
        return std_pool**2
    except TypeError:
        # Fallback to analytical KRR variance
        if hasattr(model, "_generate_kernel") and hasattr(model, "reg"):
            return _gpr_variance(model, X_train, X_pool)
        else:
            raise TypeError(
                "Estimator does not support 'return_std=True' and is not a recognized KRR class."
            )


def _uq_ensemble(
    base_estimator: object,
    X_pool: np.ndarray,
    X_train: np.ndarray,
    y_train_primary: np.ndarray,
    n_ensemble: int = 5,
) -> np.ndarray:
    """Helper for Ensemble-based uncertainty quantification."""
    pool_preds = np.zeros((X_pool.shape[0], n_ensemble), dtype=float)
    for n in range(n_ensemble):
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train_primary, train_size=0.85, random_state=n
        )
        ens_model = _train_estimator(base_estimator, X_sub, y_sub)
        pool_preds[:, n] = ens_model.predict(X_pool)

    mean_preds = np.mean(pool_preds, axis=1)
    for n in range(n_ensemble):
        pool_preds[:, n] = pool_preds[:, n] - mean_preds
    return np.sqrt(np.sum(pool_preds**2, axis=1) / (n_ensemble - 1))


def _uq_lfab(
    base_estimator: object,
    X_pool: np.ndarray,
    y_pool: Union[Dict, np.ndarray],
    X_train: np.ndarray,
    y_train: Union[Dict, np.ndarray],
) -> np.ndarray:
    """Helper for LFaB (Multi-Fidelity) uncertainty quantification."""
    if not isinstance(y_train, dict) or "low" not in y_train or "low" not in y_pool:
        raise ValueError(
            "LFaB UQ requires `y_train` and `y_pool` to be dictionaries containing a 'low' key."
        )

    model_low = _train_estimator(base_estimator, X_train, y_train["low"])
    preds_pool_low = model_low.predict(X_pool)
    return np.abs(y_pool["low"] - preds_pool_low)


def active_learning_loop(
    X_train: np.ndarray,
    y_train: Union[np.ndarray, Dict[str, np.ndarray]],
    X_pool: np.ndarray,
    y_pool: Union[np.ndarray, Dict[str, np.ndarray]],
    base_estimator: object,
    uq_mode: Union[str, Callable] = "variance",
    al_iters: int = 500,
    X_test: np.ndarray = None,
    y_test: Union[np.ndarray, Dict[str, np.ndarray]] = None,
    p_bar: bool = True,
    **uq_kwargs,
) -> dict:
    """
    Active Learning loop.

    Parameters:
    -----------
    X_train, y_train : np.ndarray or Dict
        Training data. If using 'lfab', y_train should be a dict: {'low': ..., 'high': ...}
    X_pool, y_pool : np.ndarray or Dict
        Pool data for selection.
    base_estimator : object
        ML Model providing .train() or .fit() and .predict()
    uq_mode : str or Callable
        'variance', 'ensemble', 'lfab', or a custom function: func(model, X_pool, y_pool, base_estimator, X_train, y_train, **kwargs)
    al_iters : int
        Number of AL loops to run.
    X_test, y_test : np.ndarray or Dict, optional
        If provided, MAE will be calculated and returned.
    p_bar : bool
        Show progress bar.
    **uq_kwargs : dict
        Additional arguments passed to the UQ helper (e.g. `n_ensemble=5`).
    """
    maes = []
    highest_diffs = []
    training_sizes = []
    selected_indices = []

    # Track original pool indices
    pool_tracker = np.arange(X_pool.shape[0])
    model = None

    for _ in tqdm(range(al_iters), desc=f"AL Loop ({uq_mode})", disable=not p_bar):
        training_sizes.append(X_train.shape[0])

        # Determine primary target to train the main model against
        y_train_primary = y_train["high"] if isinstance(y_train, dict) else y_train
        model = _train_estimator(base_estimator, X_train, y_train_primary)

        # Evaluate MAE if Test set is provided
        if X_test is not None and y_test is not None:
            y_test_primary = y_test["high"] if isinstance(y_test, dict) else y_test
            preds_test = model.predict(X_test)
            maes.append(np.mean(np.abs(preds_test - y_test_primary)))

        # Uncertainty Quantification
        if callable(uq_mode):
            uncertainty = uq_mode(
                model, X_pool, y_pool, base_estimator, X_train, y_train, **uq_kwargs
            )
        elif uq_mode == "variance":
            uncertainty = _uq_variance(model, X_pool, X_train)
        elif uq_mode == "ensemble":
            uncertainty = _uq_ensemble(
                base_estimator, X_pool, X_train, y_train_primary, **uq_kwargs
            )
        elif uq_mode == "lfab":
            uncertainty = _uq_lfab(base_estimator, X_pool, y_pool, X_train, y_train)
        else:
            raise ValueError(
                f"Unknown uq_mode: {uq_mode}. Use 'variance', 'ensemble', 'lfab', or pass a custom callable."
            )

        # Select highest uncertainty
        selected_idx = np.argmax(uncertainty)
        selected_indices.append(pool_tracker[selected_idx])
        highest_diffs.append(uncertainty[selected_idx])

        # Update X and ID tracker
        X_train = np.vstack((X_train, X_pool[selected_idx]))
        X_pool = np.delete(X_pool, selected_idx, axis=0)
        pool_tracker = np.delete(pool_tracker, selected_idx, axis=0)

        # Update y
        if isinstance(y_train, dict):
            for k in y_train.keys():
                y_train[k] = np.append(y_train[k], y_pool[k][selected_idx])
                y_pool[k] = np.delete(y_pool[k], selected_idx, axis=0)
        else:
            y_train = np.append(y_train, y_pool[selected_idx])
            y_pool = np.delete(y_pool, selected_idx, axis=0)

    # Return stuff as dictionary
    results = {
        "selected_indices": np.array(selected_indices),
        "training_sizes": np.array(training_sizes),
        "highest_diffs": np.array(highest_diffs),
        "model": model,
    }
    if X_test is not None and y_test is not None:
        results["maes"] = np.array(maes)

    return results
