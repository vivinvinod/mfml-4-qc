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
