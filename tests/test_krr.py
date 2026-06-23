import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal

from mfml_qc.krr import (
    matern_kernel_symmetric,
    matern_kernel_asymmetric,
    gaussian_kernel_symmetric,
    gaussian_kernel_asymmetric,
    laplacian_kernel_symmetric,
    laplacian_kernel_asymmetric,
    wasserstein_kernel_symmetric,
    wasserstein_kernel_asymmetric,
    KRR,
)

############TESTS############


@pytest.fixture
def dummy_data():
    """Generates simple dummy data for testing."""
    np.random.seed(42)
    X_train = np.random.rand(10, 3)  # 10 samples, 3 features
    y_train = np.sin(X_train[:, 0])  # Simple non-linear target
    X_test = np.random.rand(5, 3)  # 5 samples, 3 features
    return X_train, y_train, X_test


def test_matern_symmetric_properties(dummy_data):
    X_train, _, _ = dummy_data
    K = matern_kernel_symmetric(X_train, sigma=1.0, nu=1.5)

    # shape check
    assert K.shape == (10, 10)
    # symmetry check (K == K.T)
    assert_allclose(K, K.T, atol=1e-12)
    # diagonal is 1.0?
    assert_allclose(np.diag(K), np.ones(10), atol=1e-12)


def test_matern_asymmetric_shape(dummy_data):
    X_train, _, X_test = dummy_data
    K = matern_kernel_asymmetric(X_train, X_test, sigma=1.0, nu=1.5)

    # Output shape should be (n_test, n_train)
    assert K.shape == (5, 10)


def test_gaussian_symmetric_properties(dummy_data):
    X_train, _, _ = dummy_data
    K = gaussian_kernel_symmetric(X_train, sigma=2.0)

    assert K.shape == (10, 10)
    assert_allclose(K, K.T, atol=1e-12)
    assert_allclose(np.diag(K), np.ones(10), atol=1e-12)


def test_gaussian_asymmetric_shape(dummy_data):
    X_train, _, X_test = dummy_data
    K = gaussian_kernel_asymmetric(X_train, X_test, sigma=2.0)
    assert K.shape == (5, 10)


def test_laplacian_symmetric_properties(dummy_data):
    X_train, _, _ = dummy_data
    K = laplacian_kernel_symmetric(X_train, sigma=2.0)

    assert K.shape == (10, 10)
    assert_allclose(K, K.T, atol=1e-12)
    assert_allclose(np.diag(K), np.ones(10), atol=1e-12)


def test_laplacian_asymmetric_shape(dummy_data):
    X_train, _, X_test = dummy_data
    K = laplacian_kernel_asymmetric(X_train, X_test, sigma=2.0)
    assert K.shape == (5, 10)


def test_wasserstein_symmetric_properties(dummy_data):
    X_train, _, _ = dummy_data
    # Use small p, q to test robustly against NaN values
    K = wasserstein_kernel_symmetric(X_train, sigma=2.0, p=1.0, q=1.0)

    assert K.shape == (10, 10)
    assert_allclose(K, K.T, atol=1e-12)
    assert_allclose(np.diag(K), np.ones(10), atol=1e-12)


def test_wasserstein_asymmetric_shape(dummy_data):
    X_train, _, X_test = dummy_data
    K = wasserstein_kernel_asymmetric(X_train, X_test, sigma=2.0, p=1.0, q=1.0)
    assert K.shape == (5, 10)


def test_matern_unsupported_nu(dummy_data):
    X_train, _, _ = dummy_data
    # invalid nu should raise a ValueError
    with pytest.raises(ValueError, match="Only nu=0.5, 1.5, and 2.5 are supported."):
        matern_kernel_symmetric(X_train, sigma=1.0, nu=2.0)


# KRR class tests


def test_krr_initialization():
    model = KRR(kernel_type="gaussian", sigma=50.0, reg=1e-8)
    assert model.kernel_type == "gaussian"
    assert model.sigma == 50.0
    assert model.alphas is None
    # Check Wasserstein initialization
    model_wass = KRR(kernel_type="wasserstein", p=2.0, q=0.5)
    assert model_wass.p == 2.0
    assert model_wass.q == 0.5


def test_krr_predict_without_train_raises_error(dummy_data):
    _, _, X_test = dummy_data
    model = KRR()
    with pytest.raises(ValueError, match="The model has not been trained yet."):
        model.predict(X_test)


def test_krr_unsupported_kernel(dummy_data):
    X_train, y_train, _ = dummy_data
    model = KRR(kernel_type="jonathan")  # Not implemented

    with pytest.raises(ValueError, match="Unsupported kernel type"):
        model.train(X_train, y_train)


def test_krr_training_and_interpolation(dummy_data):
    X_train, y_train, _ = dummy_data

    # Train the model with very low regularization
    model = KRR(kernel_type="matern", sigma=1.0, reg=1e-10)
    model.train(X_train, y_train)

    # Assert state is updated
    assert model.alphas is not None
    assert model.alphas.shape == (10,)
    assert_equal(model.X_train, X_train)

    # KRR should predict the training data almost perfectly for very small reg
    y_pred_train = model.predict(X_train)
    assert_allclose(y_pred_train, y_train, atol=1e-5)
