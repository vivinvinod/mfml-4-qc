import numpy as np
from tqdm.auto import tqdm
import os


def property_differences(file_paths: list):
    """
    Helper utility to parse and align nested multi-fidelity datasets from raw text files.

    This function loads data from multiple fidelities and builds a strict nested
    index mapping. Crucially, it returns the *full* property values for each level
    (not the physical deltas), along with index arrays mapping each higher-fidelity
    sample back to its corresponding baseline geometry ID.

    Parameters
    ----------
    file_paths : list of str
        List of exact file paths to the energy/property files.
        Each file should be formatted with 2 columns: [timestamp/ID, property_value].
        The list MUST be ordered from the lowest fidelity (baseline) to the highest.

    Returns
    -------
    tuple
        A tuple containing `(energy_array, index_array)`:

        - **energy_array** (*np.ndarray*): A 1D object array of length `num_fidelities`.
          Each element is a 1D NumPy float array of the extracted property values for that fidelity.
        - **index_array** (*np.ndarray*): A 1D object array of length `num_fidelities`.
          Each element is a 2D NumPy integer array of shape `(N, 2)`. The columns
          represent `[baseline_row_index, current_fidelity_row_index]`.

    Raises
    ------
    FileNotFoundError
        If the baseline file (the first path in `file_paths`) cannot be located.
    """
    num_fidelities = len(file_paths)
    energy_array = np.zeros((num_fidelities), dtype=object)
    index_array = np.zeros((num_fidelities), dtype=object)

    # Load lowest fidelity file (baseline)
    if not os.path.exists(file_paths[0]):
        raise FileNotFoundError(f"Could not find baseline file at {file_paths[0]}")

    E0 = np.loadtxt(file_paths[0])
    energy_array[0] = E0[:, 1]

    # Baseline index is just 1-to-1 against itself
    index_array[0] = np.asarray(
        [np.arange(0, energy_array[0].shape[0]), np.arange(0, energy_array[0].shape[0])]
    ).T

    # O(1) Lookup dictionary mapping timestamp -> index in E0
    E0_map = {val: idx for idx, val in enumerate(E0[:, 0])}

    for i in tqdm(
        range(0, num_fidelities - 1),
        desc="Generating property array and indexes for MFML...",
        leave=False,
    ):
        Ei = np.loadtxt(file_paths[i])
        Eip1 = np.loadtxt(file_paths[i + 1])

        index = []
        # quick-lookup set for Ei timestamps
        Ei_set = set(Ei[:, 0])

        for k, val in enumerate(Eip1[:, 0]):
            # If the timestamp exists in both Ei and E0
            if val in Ei_set and val in E0_map:
                index.append([E0_map[val], k])

        index_array[i + 1] = np.asarray(index, dtype=int)
        energy_array[i + 1] = np.copy(Eip1[:, 1])

    return energy_array, index_array


def build_hierarchy_arrays(data_train: np.ndarray, hierarchy_cols: list) -> tuple:
    """
    Extracts valid subsets and mean-centers energies across a fidelity hierarchy.

    Parameters
    ----------
    data_train : np.ndarray
        The training data array containing all fidelities.
    hierarchy_cols : list of int
        Column indices for the fidelities, ordered from lowest to highest.

    Returns
    -------
    tuple
        (y_trains, indexes, means) where y_trains and indexes are object arrays
        formatted for the ModelMFML, and means are the centering offsets.
    """
    num_fids = len(hierarchy_cols)
    y_trains = np.zeros(num_fids, dtype=object)
    indexes = np.zeros(num_fids, dtype=object)
    means = np.zeros(num_fids)

    # 1. Process Baseline (Lowest Fidelity)
    baseline_col = hierarchy_cols[0]
    baseline_vals = data_train[:, baseline_col]
    means[0] = np.mean(baseline_vals)

    # Baseline is mapped 1-to-1 against itself
    y_trains[0] = baseline_vals - means[0]
    indexes[0] = np.column_stack(
        (np.arange(len(baseline_vals)), np.arange(len(baseline_vals)))
    )

    # 2. Process Higher Fidelities
    for i in range(1, num_fids):
        target_col = hierarchy_cols[i]
        valid_rows = ~np.isnan(data_train[:, target_col])

        target_vals = data_train[valid_rows, target_col]
        means[i] = np.mean(target_vals)
        y_trains[i] = target_vals - means[i]

        baseline_idx = np.where(valid_rows)[0]
        level_idx = np.arange(len(target_vals))
        indexes[i] = np.column_stack((baseline_idx, level_idx))

    return y_trains, indexes, means


def top_down_subsetting(
    y_trains: np.ndarray, indexes: np.ndarray, n_trains_target: list, seed: int = 42
) -> tuple:
    """
    Function to produce nested subsets of data from a multifidelity dataset where
    sample selection is carried out from the highest fidelity to the lowest.

    Parameters
    ----------
    y_trains : np.ndarray
        The full target properties array.
    indexes : np.ndarray
        The full mapping indexes array.
    n_trains_target : list of int
        Target number of samples for each fidelity (lowest to highest).
    seed : int, optional
        Random state seed for shuffling. Defaults to 42.

    Returns
    -------
    tuple
        (subset_y_trains, subset_indexes) ready for MFML training.
    """
    num_fids = len(y_trains)
    rng = np.random.RandomState(seed)

    subset_y_trains = np.zeros(num_fids, dtype=object)
    subset_indexes = np.zeros(num_fids, dtype=object)

    # Track baseline IDs, cascading downwards from highest to lowest
    selected_b_ids = []

    for i in range(num_fids - 1, -1, -1):
        needed = n_trains_target[i] - len(selected_b_ids)
        avail_b_ids = indexes[i][:, 0]

        selected_set = set(selected_b_ids)
        candidates = [b for b in avail_b_ids if b not in selected_set]

        if needed > 0:
            rng.shuffle(candidates)
            selected_b_ids.extend(candidates[:needed])

        # Map back to fetch original target values
        fid_map = {
            row[0]: (row[1], y_trains[i][idx]) for idx, row in enumerate(indexes[i])
        }

        extracted_data = []
        for b_idx in selected_b_ids:
            if b_idx in fid_map:
                old_lvl_idx, y_val = fid_map[b_idx]
                extracted_data.append((b_idx, y_val))

        extracted_data.sort(key=lambda x: x[0])

        final_ind = []
        final_y = []
        for new_lvl_idx, (b_idx, y_val) in enumerate(extracted_data):
            final_ind.append([b_idx, new_lvl_idx])
            final_y.append(y_val)

        subset_indexes[i] = np.array(final_ind, dtype=int)
        subset_y_trains[i] = np.array(final_y, dtype=np.float64)

    return subset_y_trains, subset_indexes
