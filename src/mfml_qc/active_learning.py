import numpy as np
import copy
from sklearn.model_selection import train_test_split
from .krr import KRR


def _train_estimator(base_estimator: object, X: np.ndarray, y: np.ndarray) -> object:
    """
    Helper function to securely instantiate and train a machine learning model.

    Uses duck typing to support arbitrary model architectures. The provided
    estimator must implement either a standard scikit-learn `.fit()` method
    or a custom `.train()` method.

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


def gpr_variance(
    X_train: np.ndarray,
    X_pool: np.ndarray,
    kernel_type: str = "matern",
    sigma: float = 100.0,
    reg: float = 1e-9,
    nu: float = 1.5,
    p: float = 1.0,
    q: float = 1.0,
) -> np.ndarray:
    r"""
    Computes the exact analytical predictive variance for a set of pool geometries.

    Because the predictive variance in Gaussian Process Regression (or Kernel Ridge
    Regression) depends strictly on the spatial distribution of the feature data and
    the kernel, it is completely independent of the target properties or model weights.

    The predictive variance for a set of pool points :math:`x_*` is given by:

    .. math::
        \mathbb{V}[x_*] = k(x_*, x_*) - k(X, x_*)^T (K(X, X) + \lambda I)^{-1} k(X, x_*)

    Parameters
    ----------
    X_train : np.ndarray
        The training feature matrix :math:`X`.
    X_pool : np.ndarray
        The pool feature matrix containing the points to evaluate :math:`x_*`.
    kernel_type : str, optional
        The type of kernel to use ('matern', 'gaussian', 'laplacian', or 'wasserstein'). Defaults to 'matern'.
    sigma : float, optional
        The kernel width parameter. Defaults to 100.0.
    reg : float, optional
        The regularization parameter (:math:`\lambda`). Defaults to 1e-9.
    nu : float, optional
        Smoothness parameter for the Matérn kernel. Defaults to 1.5.
    p : float, optional
        Power parameter for the Wasserstein kernel. Defaults to 1.0.
    q : float, optional
        Outer exponent parameter for the Wasserstein kernel. Defaults to 1.0.

    Returns
    -------
    np.ndarray
        A 1D array of computed predictive variances for each point in the pool.
    """
    dummy_model = KRR(kernel_type=kernel_type, sigma=sigma, nu=nu, p=p, q=q, reg=reg)

    # training kernel
    K_train = dummy_model._generate_kernel(X_train)
    K_train[np.diag_indices_from(K_train)] += reg

    # pool kernel
    K_pool_train = dummy_model._generate_kernel(X_train=X_train, X_test=X_pool)

    k_pool_pool_diag = 1.0

    # (K(X, X) + reg * I)^{-1} k(X, x_*)
    inv_K_train_dot_K_pt = np.linalg.solve(K_train, K_pool_train.T)

    variance = k_pool_pool_diag - np.sum(K_pool_train * inv_K_train_dot_K_pt.T, axis=1)
    return variance


def ensemble_variance(
    base_estimator: object,
    X_pool: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_ensemble: int = 5,
    train_size: float = 0.85,
) -> np.ndarray:
    """
    Computes Ensemble-based uncertainty quantification.

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
    y_train : np.ndarray
        The complete training target array.
    n_ensemble : int, optional
        The number of models to include in the ensemble. Defaults to 5.
    train_size : float, optional
        The fraction of training data to use for the ensemble of models. Default is 85 (that is 85%).

    Returns
    -------
    np.ndarray
        A 1D array of ensemble standard deviations for the pool data.
    """
    pool_preds = np.zeros((X_pool.shape[0], n_ensemble), dtype=float)
    for n in range(n_ensemble):
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train, train_size=train_size, random_state=n
        )
        ens_model = _train_estimator(base_estimator, X_sub, y_sub)
        pool_preds[:, n] = ens_model.predict(X_pool)

    mean_preds = np.mean(pool_preds, axis=1)
    for n in range(n_ensemble):
        pool_preds[:, n] = pool_preds[:, n] - mean_preds

    return np.sum(pool_preds**2, axis=1) / (n_ensemble - 1)


def lfab(
    base_estimator: object,
    X_pool: np.ndarray,
    y_pool_low: np.ndarray,
    X_train: np.ndarray,
    y_train_low: np.ndarray,
) -> np.ndarray:
    r"""
    Computes Low Fidelity as Bias (LFaB) uncertainty quantification.

    Estimates uncertainty by using the absolute prediction error of a model
    trained on low-fidelity data. Areas where the low-fidelity surrogate struggles
    are assumed to be areas of high uncertainty.

    The LFaB error for a set of pool points :math:`x_*` is computed as:

    .. math::
        LFaB[x_*] = \left\lvert \hat{P}^{low}_{ML} - P^{low}_{ref} \right\rvert

    where :math:`\hat{P}^{low}_{ML}` is the low-fidelity prediction of the ML model
    and :math:`P^{low}_{ref}` is the reference low-fidelity value.

    Parameters
    ----------
    base_estimator : object
        The machine learning model to train on the baseline data.
    X_pool : np.ndarray
        The pool feature matrix to evaluate.
    y_pool_low : np.ndarray
        The low-fidelity target data corresponding to the pool geometries.
    X_train : np.ndarray
        The training feature matrix.
    y_train_low : np.ndarray
        The low-fidelity target data corresponding to the training geometries.

    Returns
    -------
    np.ndarray
        A 1D array of absolute prediction errors on the low-fidelity baseline.
    """
    model_low = _train_estimator(base_estimator, X_train, y_train_low)
    preds_pool_low = model_low.predict(X_pool)
    return np.abs(y_pool_low - preds_pool_low)
