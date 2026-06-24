import numpy as np
from numba import njit, prange


@njit(parallel=True, fastmath=True)
def matern_kernel_symmetric(
    X: np.ndarray, sigma: float = 100.0, nu: float = 1.5
) -> np.ndarray:
    r"""
    Function to generate the symmetric Matérn kernel matrix.
    Supports smoothness parameter :math:`\nu \in \{0.5, 1.5, 2.5\}`.
    For example, the Matérn 3/2 (:math:`\nu=1.5`) kernel entry for some :math:`x_i, x_j` is computed as:

    .. math::
        K(x_i,x_j) := \exp\left(-\frac{\sqrt{3}||x_i - x_j||_2}{\sigma}\right) \left(1 + \frac{\sqrt{3}||x_i - x_j||_2}{\sigma}\right)

    Parameters
    ----------
    X : np.ndarray
        The matrix of input features.
    sigma : float, optional
        The kernel width parameter. The default is 100.0.
    nu : float, optional
        The smoothness parameter. Supported values are 0.5, 1.5, and 2.5. The default is 1.5.

    Returns
    -------
    K : np.ndarray
        Symmetric Matérn kernel matrix for the given input X.
    """
    if nu != 0.5 and nu != 1.5 and nu != 2.5:
        raise ValueError("Only nu=0.5, 1.5, and 2.5 are supported.")

    n = X.shape[0]
    K = np.zeros((n, n), dtype=np.float64)
    sqrt3 = np.sqrt(3.0)
    sqrt5 = np.sqrt(5.0)

    for i in prange(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            dist = np.linalg.norm(X[i] - X[j])

            if nu == 0.5:
                value = np.exp(-dist / sigma)
            elif nu == 1.5:
                scaled_dist = sqrt3 * dist / sigma
                value = np.exp(-scaled_dist) * (1.0 + scaled_dist)
            else:  # nu == 2.5
                scaled_dist = sqrt5 * dist / sigma
                value = np.exp(-scaled_dist) * (
                    1.0 + scaled_dist + (scaled_dist**2) / 3.0
                )

            K[i, j] = value
            K[j, i] = value

    return K


@njit(parallel=True, fastmath=True)
def matern_kernel_asymmetric(
    X_train: np.ndarray, X_test: np.ndarray, sigma: float = 100.0, nu: float = 1.5
) -> np.ndarray:
    r"""
    Function to generate the asymmetric Matérn kernel matrix.
    Supports smoothness parameter :math:`\nu \in \{0.5, 1.5, 2.5\}`.
    For example, the Matérn 3/2 (:math:`\nu=1.5`) kernel entry for some :math:`x_i \in X_{test}, x_j \in X_{train}` is computed as:

    .. math::
        K(x_i,x_j) := \exp\left(-\frac{\sqrt{3}||x_i - x_j||_2}{\sigma}\right) \left(1 + \frac{\sqrt{3}||x_i - x_j||_2}{\sigma}\right)

    Parameters
    ----------
    X_train : np.ndarray
        The first matrix of input features (e.g., training set).
    X_test : np.ndarray
        The second matrix of input features (e.g., test set).
    sigma : float, optional
        The kernel width parameter. The default is 100.0.
    nu : float, optional
        The smoothness parameter. Supported values are 0.5, 1.5, and 2.5. The default is 1.5.

    Returns
    -------
    K : np.ndarray
        Asymmetric Matérn kernel matrix for given inputs X_train and X_test.
    """
    if nu != 0.5 and nu != 1.5 and nu != 2.5:
        raise ValueError("Only nu=0.5, 1.5, and 2.5 are supported.")

    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    K = np.zeros((n_test, n_train), dtype=np.float64)
    sqrt3 = np.sqrt(3.0)
    sqrt5 = np.sqrt(5.0)

    for i in prange(n_test):
        for j in range(n_train):
            dist = np.linalg.norm(X_test[i] - X_train[j])

            if nu == 0.5:
                value = np.exp(-dist / sigma)
            elif nu == 1.5:
                scaled_dist = sqrt3 * dist / sigma
                value = np.exp(-scaled_dist) * (1.0 + scaled_dist)
            else:  # nu == 2.5
                scaled_dist = sqrt5 * dist / sigma
                value = np.exp(-scaled_dist) * (
                    1.0 + scaled_dist + (scaled_dist**2) / 3.0
                )

            K[i, j] = value

    return K


@njit(parallel=True, fastmath=True)
def gaussian_kernel_symmetric(X: np.ndarray, sigma: float = 100.0) -> np.ndarray:
    r"""
    Function to generate the symmetric Gaussian kernel matrix for a given kernel width, :math:`\sigma`.
    The Gaussian kernel matrix entry for some :math:`x_i, x_j` is computed as:

    .. math::
        K(x_i,x_j) := \exp\left(\frac{-||x_i - x_j||_2^2}{2\sigma^2}\right)

    Parameters
    ----------
    X : np.ndarray
        The matrix of input features.
    sigma : float, optional
        The kernel width parameter. The default is 100.0.

    Returns
    -------
    K : np.ndarray
        Symmetric Gaussian kernel matrix for given input X.
    """
    n = X.shape[0]
    K = np.zeros((n, n), dtype=np.float64)
    gamma = 1.0 / (2.0 * sigma**2)

    for i in prange(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            # Using sum of squared differences directly is highly efficient in Numba
            sq_dist = np.sum((X[i] - X[j]) ** 2)
            value = np.exp(-gamma * sq_dist)
            K[i, j] = value
            K[j, i] = value

    return K


@njit(parallel=True, fastmath=True)
def gaussian_kernel_asymmetric(
    X_train: np.ndarray, X_test: np.ndarray, sigma: float = 100.0
) -> np.ndarray:
    r"""
    Function to generate the asymmetric Gaussian kernel matrix for a given kernel width, :math:`\sigma`.
    The Gaussian kernel matrix entry for some :math:`x_i \in X_{test}, x_j \in X_{train}` is computed as:

    .. math::
        K(x_i,x_j) := \exp\left(\frac{-||x_i - x_j||_2^2}{2\sigma^2}\right)

    Parameters
    ----------
    X_train : np.ndarray
        The first matrix of input features (e.g., training set).
    X_test : np.ndarray
        The second matrix of input features (e.g., test set).
    sigma : float, optional
        The kernel width parameter. The default is 100.0.

    Returns
    -------
    K : np.ndarray
        Asymmetric Gaussian kernel matrix for given inputs X_train and X_test.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    K = np.zeros((n_test, n_train), dtype=np.float64)
    gamma = 1.0 / (2.0 * sigma**2)

    for i in prange(n_test):
        for j in range(n_train):
            sq_dist = np.sum((X_test[i] - X_train[j]) ** 2)
            value = np.exp(-gamma * sq_dist)
            K[i, j] = value

    return K


@njit(parallel=True, fastmath=True)
def laplacian_kernel_symmetric(X: np.ndarray, sigma: float = 100.0) -> np.ndarray:
    r"""
    Function to generate the symmetric Laplacian kernel matrix for a given kernel width, :math:`\sigma`.
    The Laplacian kernel matrix entry for some :math:`x_i, x_j` is computed as:

    .. math::
        K(x_i,x_j) := \exp\left(\frac{-||x_i - x_j||_1}{\sigma}\right)

    Parameters
    ----------
    X : np.ndarray
        The matrix of input features.
    sigma : float, optional
        The kernel width parameter. The default is 100.0.

    Returns
    -------
    K : np.ndarray
        Symmetric Laplacian kernel matrix for given input X.
    """
    n = X.shape[0]
    K = np.zeros((n, n), dtype=np.float64)

    for i in prange(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            dist = np.sum(np.abs(X[i] - X[j]))
            value = np.exp(-dist / sigma)
            K[i, j] = value
            K[j, i] = value

    return K


@njit(parallel=True, fastmath=True)
def laplacian_kernel_asymmetric(
    X_train: np.ndarray, X_test: np.ndarray, sigma: float = 100.0
) -> np.ndarray:
    r"""
    Function to generate the asymmetric Laplacian kernel matrix for a given kernel width, :math:`\sigma`.
    The Laplacian kernel matrix entry for some :math:`x_i \in X_{test}, x_j \in X_{train}` is computed as:

    .. math::
        K(x_i,x_j) := \exp\left(\frac{-||x_i - x_j||_1}{\sigma}\right)

    Parameters
    ----------
    X_train : np.ndarray
        The first matrix of input features (e.g., training set).
    X_test : np.ndarray
        The second matrix of input features (e.g., test set).
    sigma : float, optional
        The kernel width parameter. The default is 100.0.

    Returns
    -------
    K : np.ndarray
        Asymmetric Laplacian kernel matrix for given inputs X_train and X_test.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]
    K = np.zeros((n_test, n_train), dtype=np.float64)

    for i in prange(n_test):
        for j in range(n_train):
            dist = np.sum(np.abs(X_test[i] - X_train[j]))
            value = np.exp(-dist / sigma)
            K[i, j] = value

    return K


@njit(fastmath=True)
def _wasserstein_dist(
    a_sorted: np.ndarray, b_sorted: np.ndarray, p: float, q: float
) -> float:
    r"""
    Helper function to evaluate the 1D Wasserstein distance between two sets of sorted features.

    This function calculates the Wasserstein distance using the 1D empirical
    cumulative distribution functions (CDFs) of the input arrays.

    Mathematically, the :math:`p`-Wasserstein distance between two 1D distributions
    with CDFs :math:`F_a` and :math:`F_b` is computed as:

    .. math::
        W_p(a, b) = \left( \int_{-\infty}^{\infty} |F_a(x) - F_b(x)|^p dx \right)^{1/p}

    This function then applies an outer exponent :math:`q`, returning :math:`W_p(a, b)^q`.

    Parameters
    ----------
    a_sorted : np.ndarray
        A 1D array of sorted features for the first sample.
    b_sorted : np.ndarray
        A 1D array of sorted features for the second sample.
    p : float
        The power parameter for the CDF difference (Wasserstein-p).
    q : float
        The outer exponent applied to the final distance metric.

    Returns
    -------
    float
        The calculated Wasserstein distance between the two feature sets.
    """
    m = a_sorted.shape[0]
    all_values = np.empty(2 * m, dtype=np.float64)
    all_values[:m] = a_sorted
    all_values[m:] = b_sorted
    all_values.sort()

    dist_p = 0.0
    for k in range(2 * m - 1):
        val = all_values[k]
        delta = all_values[k + 1] - val
        # Optimization: Only calculate differences where the interval > 0
        if delta > 0.0:
            a_cdf = np.searchsorted(a_sorted, val, side="right") / m
            b_cdf = np.searchsorted(b_sorted, val, side="right") / m
            dist_p += (abs(a_cdf - b_cdf) ** p) * delta

    return (dist_p ** (1.0 / p)) ** q


@njit(parallel=True, fastmath=True)
def wasserstein_kernel_symmetric(
    X: np.ndarray, sigma: float = 100.0, p: float = 1.0, q: float = 1.0
) -> np.ndarray:
    r"""
    Function to generate the symmetric Wasserstein kernel matrix based on 1D representations.

    The Wasserstein kernel matrix entry for some :math:`x_i, x_j` is computed as:

    .. math::
        K(x_i,x_j) := \exp\left(-\frac{W_p(x_i, x_j)^q}{\sigma}\right)

    where :math:`W_p` is the 1-dimensional Wasserstein distance between the sorted features.

    Parameters
    ----------
    X : np.ndarray
        The matrix of input features.
    sigma : float, optional
        The kernel width parameter. The default is 100.0.
    p : float, optional
        The power parameter for the CDF difference. The default is 1.0.
    q : float, optional
        The outer exponent parameter. The default is 1.0.

    Returns
    -------
    K : np.ndarray
        Symmetric Wasserstein kernel matrix.
    """
    n = X.shape[0]

    X_sorted = np.empty_like(X)
    for i in prange(n):
        X_sorted[i, :] = np.sort(X[i, :])

    K = np.zeros((n, n), dtype=np.float64)
    for i in prange(n):
        K[i, i] = 1.0
        for j in range(i + 1, n):
            dist = _wasserstein_dist(X_sorted[i], X_sorted[j], p, q)
            value = np.exp(-dist / sigma)
            K[i, j] = value
            K[j, i] = value

    return K


@njit(parallel=True, fastmath=True)
def wasserstein_kernel_asymmetric(
    X_train: np.ndarray,
    X_test: np.ndarray,
    sigma: float = 100.0,
    p: float = 1.0,
    q: float = 1.0,
) -> np.ndarray:
    r"""
    Function to generate the asymmetric Wasserstein kernel matrix based on 1D representations.

    The Wasserstein kernel matrix entry for some :math:`x_i \in X_{test}, x_j \in X_{train}` is computed as:

    .. math::
        K(x_i,x_j) := \exp\left(-\frac{W_p(x_i, x_j)^q}{\sigma}\right)

    where :math:`W_p` is the 1-dimensional Wasserstein distance between the sorted features.

    Parameters
    ----------
    X_train : np.ndarray
        The first matrix of input features (e.g., training set).
    X_test : np.ndarray
        The second matrix of input features (e.g., test set).
    sigma : float, optional
        The kernel width parameter. The default is 100.0.
    p : float, optional
        The power parameter for the CDF difference. The default is 1.0.
    q : float, optional
        The outer exponent parameter. The default is 1.0.

    Returns
    -------
    K : np.ndarray
        Asymmetric Wasserstein kernel matrix.
    """
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    X_train_sorted = np.empty_like(X_train)
    for i in prange(n_train):
        X_train_sorted[i, :] = np.sort(X_train[i, :])

    X_test_sorted = np.empty_like(X_test)
    for i in prange(n_test):
        X_test_sorted[i, :] = np.sort(X_test[i, :])

    K = np.zeros((n_test, n_train), dtype=np.float64)
    for i in prange(n_test):
        for j in range(n_train):
            dist = _wasserstein_dist(X_test_sorted[i], X_train_sorted[j], p, q)
            value = np.exp(-dist / sigma)
            K[i, j] = value

    return K


class KRR:
    """
    Kernel Ridge Regression (KRR) model.

    This class implements a lightweight, pure-NumPy KRR solver tailored
    for the ultra-fast Numba-compiled kernels provided in this module.
    """

    def __init__(
        self,
        kernel_type: str = "matern",
        sigma: float = 100.0,
        nu: float = 1.5,
        p: float = 1.0,
        q: float = 1.0,
        reg: float = 1e-10,
    ):
        """
        Initializes the KRR model.

        Parameters
        ----------
        kernel_type : str, optional
            The type of kernel to use ('matern', 'gaussian', 'laplacian', or 'wasserstein').
            The default is 'matern'.
        sigma : float, optional
            The kernel width parameter. The default is 100.0.
        nu : float, optional
            The smoothness parameter for the Matérn kernel. The default is 1.5.
        p : float, optional
            The power parameter for the Wasserstein kernel. The default is 1.0.
        q : float, optional
            The outer exponent parameter for the Wasserstein kernel. The default is 1.0.
        reg : float, optional
            The regularization parameter (lambda) added to the diagonal of the
            kernel matrix to prevent overfitting and singular matrices. The default is 1e-10.
        """
        self.kernel_type = kernel_type
        self.sigma = sigma
        self.nu = nu
        self.p = p
        self.q = q
        self.reg = reg
        self.alphas = None
        self.X_train = None  # we need that again for test kernel

    def _generate_kernel(
        self, X_train: np.ndarray, X_test: np.ndarray = None
    ) -> np.ndarray:
        """
        Helper method to generate the requested kernel matrix.

        Parameters
        ----------
        X_train : np.ndarray
            The training feature matrix.
        X_test : np.ndarray, optional
            The testing feature matrix. If provided, generates an asymmetric kernel.
            If None, generates a symmetric training kernel. The default is None.

        Returns
        -------
        np.ndarray
            The evaluated kernel matrix.
        """
        if self.kernel_type == "matern":
            if X_test is None:
                return matern_kernel_symmetric(X_train, sigma=self.sigma, nu=self.nu)
            else:
                return matern_kernel_asymmetric(
                    X_train, X_test, sigma=self.sigma, nu=self.nu
                )
        elif self.kernel_type == "gaussian":
            if X_test is None:
                return gaussian_kernel_symmetric(X_train, sigma=self.sigma)
            else:
                return gaussian_kernel_asymmetric(X_train, X_test, sigma=self.sigma)
        elif self.kernel_type == "laplacian":
            if X_test is None:
                return laplacian_kernel_symmetric(X_train, sigma=self.sigma)
            else:
                return laplacian_kernel_asymmetric(X_train, X_test, sigma=self.sigma)
        elif self.kernel_type == "wasserstein":
            if X_test is None:
                return wasserstein_kernel_symmetric(
                    X_train, sigma=self.sigma, p=self.p, q=self.q
                )
            else:
                return wasserstein_kernel_asymmetric(
                    X_train, X_test, sigma=self.sigma, p=self.p, q=self.q
                )
        else:
            raise ValueError(
                f"Unsupported kernel type: '{self.kernel_type}'. Use 'matern', 'gaussian', 'laplacian', or 'wasserstein'."
            )

    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """
        Fits the Kernel Ridge Regression model to the training data.

        Parameters
        ----------
        X_train : np.ndarray
            The training feature data matrix.
        y_train : np.ndarray
            The training target data array.
        """
        # Store training features as they are required for prediction
        self.X_train = np.copy(X_train)

        # Generate training kernel matrix
        k_train = self._generate_kernel(X_train=self.X_train)

        # Apply regularization to the diagonal
        k_train[np.diag_indices_from(k_train)] += self.reg

        # Solve for coefficients (alphas)
        self.alphas = np.linalg.solve(k_train, y_train)

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Predicts target values for the given test data.

        Parameters
        ----------
        X_test : np.ndarray
            The testing feature data matrix.

        Returns
        -------
        np.ndarray
            The predicted target values.

        Raises
        ------
        ValueError
            If the model has not been trained yet.
        """
        if self.alphas is None or self.X_train is None:
            raise ValueError("The model has not been trained yet. Call .train() first.")

        # Generate test kernel matrix against stored training data
        k_test = self._generate_kernel(X_train=self.X_train, X_test=X_test)

        # Compute predictions
        # k_test has shape (n_test, n_train) and self.alphas has shape (n_train,)
        return np.dot(k_test, self.alphas)
