import pytest
import numpy as np
from sklearn.linear_model import Ridge

from mfml_qc.mfml import ModelMFML


@pytest.fixture
def mfml_synthetic_data():
    """
    Creates a perfectly controlled 2-fidelity dataset.
    Fidelity 0 (Low): 10 samples
    Fidelity 1 (High): 5 samples (Nested subset of Fidelity 0)
    """
    np.random.seed(42)
    # 10 parent features
    X_train_parent = np.random.rand(10, 3)

    # 10 low fidelity properties
    y_low = np.random.rand(10)

    # 5 high fidelity properties
    y_high = np.random.rand(5)

    # Baseline index is 1-to-1: [Parent_ID, Prop_ID]
    index_low = np.array([[i, i] for i in range(10)])

    # High fidelity index: uses parent IDs 0, 2, 4, 6, 8
    index_high = np.array([[0, 0], [2, 1], [4, 2], [6, 3], [8, 4]])

    y_trains = np.array([y_low, y_high], dtype=object)
    indexes = np.array([index_low, index_high], dtype=object)

    return X_train_parent, y_trains, indexes


def test_nested_indexing(mfml_synthetic_data):
    """Tests if the nested subsetting correctly limits data without breaking links."""
    _, y_trains, indexes = mfml_synthetic_data

    # Initialize model
    model = ModelMFML()
    model.indexes = indexes
    model.y_trains = y_trains

    # Request exactly 6 low-fidelity and 3 high-fidelity samples
    n_trains = np.array([6, 3])

    # Shuffle = False -> It should grab the first available ones sequentially
    subset_indexes = model._generate_nested_indexes(n_trains=n_trains, shuffle=False)

    # Check lengths
    assert len(subset_indexes[0]) == 6
    assert len(subset_indexes[1]) == 3

    # Check nesting guarantee: Every Parent_ID in high MUST exist in low
    low_parent_ids = set(subset_indexes[0][:, 0])
    for high_parent_id in subset_indexes[1][:, 0]:
        assert (
            high_parent_id in low_parent_ids
        ), f"Parent ID {high_parent_id} missing from baseline!"


def test_mfml_training_mechanics(mfml_synthetic_data):
    """Tests if ModelMFML correctly maps 2 fidelities to 3 sub-models."""
    X_train_parent, y_trains, indexes = mfml_synthetic_data

    # Ridge() as a dummy base estimator so tests run instantly
    model = ModelMFML(base_estimator=Ridge())

    # Train using the arrays
    model.train(X_train_parent=X_train_parent, y_trains=y_trains, indexes=indexes)

    # For N=2 fidelities, it should train exactly 2N - 1 = 3 models
    assert model.models is not None
    assert len(model.models) == 3

    # Ensure all 3 models actually got trained
    for sub_model in model.models:
        # scikit-learn Ridge models will have 'coef_' after being fit
        assert hasattr(sub_model, "coef_")


def test_mfml_prediction_and_optimizers(mfml_synthetic_data):
    """Tests default SGCT prediction and post-training LRR optimization."""
    X_train_parent, y_trains, indexes = mfml_synthetic_data

    # Train
    model = ModelMFML(base_estimator=Ridge())
    model.train(X_train_parent=X_train_parent, y_trains=y_trains, indexes=indexes)

    # fake test data
    X_test = np.random.rand(4, 3)

    # Default
    preds_default = model.predict(X_test=X_test, optimiser="default")
    assert preds_default.shape == (4,)

    # LRR Optimizer
    # Requires a validation set to learn the coefficients
    X_val = np.random.rand(5, 3)
    y_val = np.random.rand(5)

    preds_lrr = model.predict(X_test=X_test, X_val=X_val, y_val=y_val, optimiser="LRR")
    assert preds_lrr.shape == (4,)
    # Verify the LCCoptimizer got saved
    assert hasattr(model.LCCoptimizer, "coef_")
