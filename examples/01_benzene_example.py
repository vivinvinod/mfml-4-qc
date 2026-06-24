"""
Benzene Multi-Fidelity Example
==============================

This example demonstrates how to use the MFML-QC package to load the
built-in Benzene trajectory dataset, manually extract a
multi-fidelity subset using a top-down approach, and train an MFML model
to predict high-fidelity excitation energies.
"""

import os
import numpy as np
from mfml_qc.datasets import load_benzene_data
from mfml_qc.mfml import ModelMFML


def main():
    print("Loading Benzene dataset via built-in loader...")
    dataset = load_benzene_data()

    X_CM = dataset["X_CM"]
    data = dataset["energies"]

    # 4 fidelity example: LC-DFTB (Col 2) -> STO-3G (Col 3) -> def2-SVP (Col 6) -> def2-TZVP (Col 7)
    hierarchy_cols = [2, 3, 6, 7]
    num_fids = len(hierarchy_cols)

    # Split into Train (0 to 12287) and Test (12288 to 14999)
    train_mask = data[:, 0] < 12288
    test_mask = data[:, 0] >= 12288

    X_train_parent = X_CM[train_mask]
    X_test = X_CM[test_mask]
    data_train = data[train_mask]
    data_test = data[test_mask]

    ### MFML dataset formulation
    print("Extracting nested data")
    y_trains = np.zeros(num_fids, dtype=object)
    indexes = np.zeros(num_fids, dtype=object)
    means = np.zeros(num_fids)

    # Baseline fidelity (LC-DFTB)
    baseline_col = hierarchy_cols[0]
    baseline_vals = data_train[:, baseline_col]
    means[0] = np.mean(baseline_vals)

    # LC-DFTB has no NaNs or missing entries so we just use it all
    y_trains[0] = baseline_vals - means[0]  # center data
    indexes[0] = np.column_stack(
        (np.arange(len(baseline_vals)), np.arange(len(baseline_vals)))
    )
    print(f"Fidelity 0 has {len(baseline_vals)} training samples.")
    # Higher fidelities
    for i in range(1, num_fids):
        target_col = hierarchy_cols[i]
        # Find rows where the higher fidelity calculation actually exists (not NaN)
        valid_rows = ~np.isnan(data_train[:, target_col])

        # Extract energies
        target_vals = data_train[valid_rows, target_col]
        means[i] = np.mean(target_vals)
        y_trains[i] = target_vals - means[i]

        # Build index mapping: [Baseline_Row_ID, HighFid_Row_ID]
        baseline_idx = np.where(valid_rows)[0]
        level_idx = np.arange(len(target_vals))
        indexes[i] = np.column_stack((baseline_idx, level_idx))

        print(f"Fidelity {i} has {len(indexes[i])} training samples.")

    # fix n_trains
    # since benzene data is sparse populated, we will do a top down approach
    # for the training data sampling
    n_trains_target = np.asarray([1024, 512, 256, 128])
    rng = np.random.RandomState(42)

    subset_y_trains = np.zeros(num_fids, dtype=object)
    subset_indexes = np.zeros(num_fids, dtype=object)

    # Track the baseline IDs we select, going downwards from highest to lowest fidelity
    selected_b_ids = []

    for i in range(num_fids - 1, -1, -1):
        needed = n_trains_target[i] - len(selected_b_ids)
        avail_b_ids = indexes[i][:, 0]
        # Consider baseline IDs that aren't already selected by a higher fidelity
        selected_set = set(selected_b_ids)
        candidates = [b for b in avail_b_ids if b not in selected_set]
        if needed > 0:
            rng.shuffle(candidates)
            selected_b_ids.extend(candidates[:needed])
        # Quick mapping to fetch the original target values
        fid_map = {
            row[0]: (row[1], y_trains[i][idx]) for idx, row in enumerate(indexes[i])
        }

        # Extract the specific rows for the baseline IDs we've accumulated so far
        extracted_data = []
        for b_idx in selected_b_ids:
            if b_idx in fid_map:
                old_lvl_idx, y_val = fid_map[b_idx]
                extracted_data.append((b_idx, y_val))

        extracted_data.sort(key=lambda x: x[0])

        # Rebuild indexes and y_trains so the level_idx matches the new truncated arrays
        final_ind = []
        final_y = []
        for new_lvl_idx, (b_idx, y_val) in enumerate(extracted_data):
            final_ind.append([b_idx, new_lvl_idx])
            final_y.append(y_val)

        subset_indexes[i] = np.array(final_ind, dtype=int)
        subset_y_trains[i] = np.array(final_y, dtype=np.float64)

        print(f"Fidelity {i} explicitly truncated to {len(final_y)} nested samples.")

    print("\nInitializing MFML Model")
    mfml_model = ModelMFML(kernel="matern", sigma=715.0, reg=1e-9, p_bar=True)

    print(f"Training MFML with training data size: {n_trains_target}")
    mfml_model.train(X_train_parent=X_train_parent, y_trains=y_trains, indexes=indexes)

    # pick def2-TZVP test data
    y_test_true = data_test[:, hierarchy_cols[-1]]

    print("\nPredicting on Test Set (2712 samples) with default MFML")
    preds = mfml_model.predict(X_test=X_test, optimiser="default")
    preds += means[-1]  # everything was centered. So uncenter by mean.

    # MAE computation
    mae = np.mean(np.abs(preds - y_test_true))
    mae_kcal = mae * 23

    print(f"MFML Test Set MAE: {mae:.6f} eV ({mae_kcal:.4f} kcal/mol)")


if __name__ == "__main__":
    main()
