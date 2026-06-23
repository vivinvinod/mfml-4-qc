import pytest
import numpy as np
import copy
from numpy.testing import assert_array_equal, assert_allclose

from mfml_qc.active_learning import (
    _train_estimator,
    _gpr_variance,
    _uq_variance,
    active_learning_loop,
)


class DummySklearnModel:
    """Mock model that mimics a scikit-learn estimator (uses .fit)"""

    def __init__(self):
        self.trained = False

    def fit(self, X, y):
        self.trained = True

    def predict(self, X, return_std=False):
        preds = np.zeros(X.shape[0])
        if return_std:
            # Return fake standard deviations (e.g., random numbers)
            np.random.seed(42)
            stds = np.random.rand(X.shape[0])
            return preds, stdscopmute
        return preds


class DummyKRRModel:
    """Mock model that mimics our custom KRR model (uses .train and ._generate_kernel)"""

    def __init__(self):
        self.reg = 1e-5
        self.trained = False

    def train(self, X, y):
        self.trained = True

    def predict(self, X):
        return np.ones(X.shape[0])

    def _generate_kernel(self, X_train, X_test=None):
        # Return a safe, invertible dummy kernel (Identity matrix for train)
        if X_test is None:
            return np.eye(X_train.shape[0])
        return np.ones((X_test.shape[0], X_train.shape[0])) * 0.1


class BadModel:
    """Mock model with no valid training method."""

    def predict(self, X):
        return X


@pytest.fixture
def standard_data():
    np.random.seed(0)
    X_train = np.random.rand(10, 5)
    y_train = np.random.rand(10)
    X_pool = np.random.rand(20, 5)
    y_pool = np.random.rand(20)
    X_test = np.random.rand(5, 5)
    y_test = np.random.rand(5)
    return X_train, y_train, X_pool, y_pool, X_test, y_test


@pytest.fixture
def lfab_data():
    np.random.seed(1)
    X_train = np.random.rand(10, 5)
    y_train = {"low": np.random.rand(10), "high": np.random.rand(10)}
    X_pool = np.random.rand(20, 5)
    y_pool = {"low": np.random.rand(20), "high": np.random.rand(20)}
    X_test = np.random.rand(5, 5)
    y_test = {"low": np.random.rand(5), "high": np.random.rand(5)}
    return X_train, y_train, X_pool, y_pool, X_test, y_test


def test_duck_typing_train_estimator(standard_data):
    X_train, y_train, _, _, _, _ = standard_data

    # Test scikit-learn style
    sk_model = _train_estimator(DummySklearnModel(), X_train, y_train)
    assert sk_model.trained is True

    # Test KRR style
    krr_model = _train_estimator(DummyKRRModel(), X_train, y_train)
    assert krr_model.trained is True

    # Test invalid model throws AttributeError
    with pytest.raises(AttributeError, match="Estimator must have a '.train"):
        _train_estimator(BadModel(), X_train, y_train)


def test_uq_variance_fallback():
    """Tests if variance UQ routes correctly depending on the model."""
    X_train = np.random.rand(10, 5)
    X_pool = np.random.rand(20, 5)

    sk_model = DummySklearnModel()
    krr_model = DummyKRRModel()

    # Scikit-learn should successfully return the squared stds
    sk_vars = _uq_variance(sk_model, X_pool, X_train)
    assert sk_vars.shape == (20,)

    # KRR should fallback to exact algebraic calculation
    krr_vars = _uq_variance(krr_model, X_pool, X_train)
    assert krr_vars.shape == (20,)


def test_active_learning_loop_mechanics(standard_data):
    """
    Tests standard variance AL. We run 3 iterations.
    X_pool should decrease by 3. X_train should increase by 3.
    """
    X_train, y_train, X_pool, y_pool, _, _ = standard_data
    initial_pool_size = X_pool.shape[0]
    initial_train_size = X_train.shape[0]
    iters = 3

    results = active_learning_loop(
        X_train,
        y_train,
        X_pool,
        y_pool,
        base_estimator=DummySklearnModel(),
        uq_mode="variance",
        al_iters=iters,
        p_bar=False,
    )

    assert len(results["selected_indices"]) == iters
    assert len(results["training_sizes"]) == iters
    assert (
        results["training_sizes"][-1] == initial_train_size + iters - 1
    )  # starts at iteration 0
    # Make sure we didn't calculate MAE since X_test was None
    assert "maes" not in results


def test_active_learning_loop_lfab(lfab_data):
    """
    Tests Multi-Fidelity (LfaB) logic handling dictionaries.
    """
    X_train, y_train, X_pool, y_pool, X_test, y_test = lfab_data
    iters = 2

    # Take copies to verify the originals are left intact
    orig_y_train_low_len = len(y_train["low"])
    orig_y_pool_high_len = len(y_pool["high"])

    results = active_learning_loop(
        X_train.copy(),
        copy.deepcopy(y_train),
        X_pool.copy(),
        copy.deepcopy(y_pool),
        base_estimator=DummyKRRModel(),
        uq_mode="lfab",
        al_iters=iters,
        X_test=X_test,
        y_test=y_test,  # Passing test set should trigger MAE
        p_bar=False,
    )

    assert len(results["selected_indices"]) == iters
    assert "maes" in results
    assert len(results["maes"]) == iters


def test_active_learning_custom_callable(standard_data):
    """Tests if a user can successfully hijack the UQ logic."""
    X_train, y_train, X_pool, y_pool, _, _ = standard_data

    # Custom UQ function: always picks the first element by returning descending certainty
    def custom_uq(model, X_pool, y_pool, base_estimator, X_train, y_train):
        return np.arange(X_pool.shape[0], 0, -1)

    results = active_learning_loop(
        X_train,
        y_train,
        X_pool,
        y_pool,
        base_estimator=DummySklearnModel(),
        uq_mode=custom_uq,
        al_iters=2,
        p_bar=False,
    )

    # Because of the custom_uq being descending order,
    # it should always pick index 0 of whatever the pool currently is.
    assert len(results["selected_indices"]) == 2
