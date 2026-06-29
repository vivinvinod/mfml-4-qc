import pytest
import numpy as np
from numpy.testing import assert_allclose
from sklearn.linear_model import Ridge

from mfml_qc.active_learning import (
    _train_estimator,
    gpr_variance,
    ensemble_variance,
    lfab,
)

# --- Dummy Models for Duck Typing ---


class DummySklearnModel:
    """Mock model that mimics a scikit-learn estimator (uses .fit)"""

    def __init__(self):
        self.trained = False

    def fit(self, X, y):
        self.trained = True
        return self

    def predict(self, X):
        return np.zeros(X.shape[0])


class DummyKRRModel:
    """Mock model that mimics our custom KRR model (uses .train)"""

    def __init__(self):
        self.trained = False

    def train(self, X, y):
        self.trained = True
        return self

    def predict(self, X):
        return np.ones(X.shape[0])


class BadModel:
    """Mock model with no valid training method."""

    def predict(self, X):
        return np.zeros(X.shape[0])


# --- Fixtures ---


@pytest.fixture
def uq_data():
    """Provides standard dataset shapes for UQ calculations."""
    np.random.seed(42)
    # 10 training samples, 20 pool samples, 5 features
    X_train = np.random.rand(10, 5)
    y_train = np.random.rand(10)
    X_pool = np.random.rand(20, 5)
    y_pool = np.random.rand(20)

    return X_train, y_train, X_pool, y_pool


# --- Tests ---


def test_duck_typing_train_estimator(uq_data):
    """Tests if the estimator training helper correctly identifies duck typed models."""
    X_train, y_train, _, _ = uq_data

    # Test scikit-learn style (.fit)
    sk_model = _train_estimator(DummySklearnModel(), X_train, y_train)
    assert sk_model.trained is True

    # Test KRR style (.train)
    krr_model = _train_estimator(DummyKRRModel(), X_train, y_train)
    assert krr_model.trained is True

    # Test invalid model throws AttributeError
    with pytest.raises(AttributeError, match="Estimator must have a '.train"):
        _train_estimator(BadModel(), X_train, y_train)


def test_compute_gpr_variance(uq_data):
    """Tests the exact algebraic Gaussian Process/KRR variance calculation."""
    X_train, _, X_pool, _ = uq_data

    variances = gpr_variance(
        X_train=X_train, X_pool=X_pool, kernel_type="matern", sigma=10.0, reg=1e-6
    )

    # 1. Check shape matches the pool size
    assert variances.shape == (20,)

    # 2. Check bounds: GPR variance should always be positive and theoretically <= 1.0
    # (since k(x_*, x_*) = 1.0 for our stationary kernels)
    assert np.all(variances >= 0.0)
    assert np.all(variances <= 1.0 + 1e-12)  # Small numerical tolerance


def test_compute_gpr_variance_zero_distance(uq_data):
    """Tests that evaluating GPR variance on the exact training data yields near-zero variance."""
    X_train, _, _, _ = uq_data

    # If the pool IS the training data, variance should be extremely small (bounded by reg)
    variances = gpr_variance(
        X_train=X_train, X_pool=X_train, kernel_type="gaussian", sigma=1.0, reg=1e-9
    )

    assert variances.shape == (10,)
    assert np.all(variances < 1e-5)


def test_ensemble_variance(uq_data):
    """Tests the ensemble-based uncertainty quantification."""
    X_train, y_train, X_pool, _ = uq_data

    # Use an actual scikit-learn model so we get non-zero variances
    base_model = Ridge(alpha=1.0)
    n_ensemble = 5

    ens_var = ensemble_variance(
        base_estimator=base_model,
        X_pool=X_pool,
        X_train=X_train,
        y_train=y_train,
        n_ensemble=n_ensemble,
    )

    # 1. Check shape
    assert ens_var.shape == (20,)

    # 2. Check bounds: Variance must be >= 0
    assert np.all(ens_var >= 0.0)


def test_lfab(uq_data):
    """Tests the Low Fidelity as Bias (LFaB) uncertainty quantification."""
    X_train, y_train_low, X_pool, y_pool_low = uq_data

    base_model = Ridge(alpha=1.0)

    lfab_errors = lfab(
        base_estimator=base_model,
        X_pool=X_pool,
        y_pool_low=y_pool_low,
        X_train=X_train,
        y_train_low=y_train_low,
    )

    # 1. Check shape
    assert lfab_errors.shape == (20,)

    # 2. Check bounds: LFaB returns absolute error, which must be >= 0
    assert np.all(lfab_errors >= 0.0)
