import numpy as np
import copy
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from typing import Union, Dict, Callable


def _train_estimator(base_estimator: object, X: np.ndarray, y: np.ndarray) -> object:
    """
    Helper function to securely instantiate and train a machine learning model.

    Uses duck typing to support arbitrary model architectures. The provided
    base estimator must implement either a `.fit()` method
    or a `.train()` method.

    Parameters
    ----------
    base_estimator : object
        The base machine learning model instance to duplicate and train.
    X : np.ndarray
        The training feature matrix.
    y : np.ndarray
        The training target array.

    Returns
    -------
    object
        A newly trained copy of the base estimator.

    Raises
    ------
    AttributeError
        If the estimator lacks both `.train()` and `.fit()` methods.
    """
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
    Computes exact predictive variance used in standard GAussian Process Regression.

    The predictive variance for a set of pool points $x_*$ is computed as:
    $$
    \\mathbb{V}[f_*] = k(x_*, x_*) - k(X, x_*)^T (K(X, X) + \\lambda I)^{-1} k(X, x_*)
    $$

    Parameters
    ----------
    model : object
        The initialized KRR model containing the `_generate_kernel` method and `reg` attribute.
    X_train : np.ndarray
        The training feature matrix $X$.
    X_pool : np.ndarray
        The pool feature matrix containing the points to evaluate $x_*$.

    Returns
    -------
    np.ndarray
        A 1D array of computed predictive variances for each point in the pool.
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
    """
    Helper function for Variance-based uncertainty quantification.

    Parameters
    ----------
    model : object
        The trained machine learning model.
    X_pool : np.ndarray
        The pool feature matrix to evaluate.
    X_train : np.ndarray
        The feature matrix the model was trained on (required for analytic fallback).

    Returns
    -------
    np.ndarray
        A 1D array of variance estimates for the pool data.

    Raises
    ------
    TypeError
        If the model does not support `return_std=True` and is not a recognized KRR model.
    """
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
    """
    Helper for Ensemble-based uncertainty quantification.

    Estimates uncertainty by training an ensemble of models on random subsets
    (85%) of the training data and computing the variance of their predictions.

    Parameters
    ----------
    base_estimator : object
        The base machine learning model to use for the ensemble.
    X_pool : np.ndarray
        The pool feature matrix to evaluate.
    X_train : np.ndarray
        The complete training feature matrix.
    y_train_primary : np.ndarray
        The complete training target array.
    n_ensemble : int, optional
        The number of models to include in the ensemble. Defaults to 5.

    Returns
    -------
    np.ndarray
        A 1D array of ensemble standard deviations for the pool data.
    """
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
    return np.sum(pool_preds**2, axis=1) / (n_ensemble - 1)


def _uq_lfab(
    base_estimator: object,
    X_pool: np.ndarray,
    y_pool: Union[Dict, np.ndarray],
    X_train: np.ndarray,
    y_train: Union[Dict, np.ndarray],
) -> np.ndarray:
    """
    Helper for Low Fidelity as Bias (LFaB) uncertainty.

    Estimates uncertainty by using the absolute prediction error of a model
    trained on a cheap, low-fidelity data.

    Parameters
    ----------
    base_estimator : object
        The machine learning model to train on the baseline data.
    X_pool : np.ndarray
        The pool feature matrix to evaluate.
    y_pool : dict or np.ndarray
        The target data for the pool. Must be a dictionary containing a 'low' key.
    X_train : np.ndarray
        The training feature matrix.
    y_train : dict or np.ndarray
        The target data for training. Must be a dictionary containing a 'low' key.

    Returns
    -------
    np.ndarray
        A 1D array of absolute prediction errors on the low-fidelity baseline.

    Raises
    ------
    ValueError
        If `y_train` or `y_pool` are missing the required 'low' dictionary keys.
    """
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
    Active Learning (AL) loop for training data sampling.

    Iteratively trains a model, evaluates the uncertainty of an unlabeled/pool
    dataset using a specified strategy, and selects the most uncertain sample
    to add to the training set for the next iteration.

    Parameters
    ----------
    X_train : np.ndarray
        The initial training feature matrix.
    y_train : np.ndarray or dict
        The initial training target data. If using 'lfab', `y_train` must
        be a dict structured as `{'low': ..., 'high': ...}`.
    X_pool : np.ndarray
        The pool feature matrix from which new samples are selected.
    y_pool : np.ndarray or dict
        The target data corresponding to the pool (revealed sequentially to the model).
    base_estimator : object
        The machine learning model providing `.train()`/`.fit()` and `.predict()`.
    uq_mode : str or callable, optional
        The uncertainty quantification strategy. Options are 'variance', 'ensemble',
        'lfab', or a custom function:
        `func(model, X_pool, y_pool, base_estimator, X_train, y_train, **kwargs)`.
        Defaults to 'variance'.
    al_iters : int, optional
        Number of active learning loops (samples to select). Defaults to 500.
    X_test : np.ndarray, optional
        The testing feature matrix used solely to track generalization error. Defaults to None.
    y_test : np.ndarray or dict, optional
        The testing target data. Defaults to None.
    p_bar : bool, optional
        Displays a tqdm progress bar if True. Defaults to True.
    **uq_kwargs : dict
        Additional arguments passed to the chosen UQ helper (e.g., `n_ensemble=5`).

    Returns
    -------
    dict
        A dictionary of the active learning results containing:
        - 'selected_indices': np.ndarray (Original global pool indices selected)
        - 'training_sizes': np.ndarray (Training set sizes per iteration)
        - 'highest_diffs': np.ndarray (Max uncertainty values per iteration)
        - 'model': object (The final trained ML model)
        - 'maes': np.ndarray (Mean Absolute Error tracking, if `X_test` was provided)
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
