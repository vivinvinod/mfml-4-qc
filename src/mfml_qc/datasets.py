import os
import numpy as np
from typing import Dict, Any
from .representations import generate_coulomb_matrices


def load_benzene_data(data_dir: str = None) -> Dict[str, Any]:
    """
    Helper utility to load and parse the built-in Benzene trajectory dataset.

    Args:
        data_dir (str, optional): Path to the benzene data directory. If None,
            it dynamically resolves relative to this installed package's directory.

    Returns:
        dict: A dictionary containing:
            - 'X_CM': Flattened Coulomb matrices (shape: 15000, 36)
            - 'energies': Raw energies CSV data (columns: Time, ZINDO, LC-DFTB, ...)
            - 'timecosts': Time costs CSV data
            - 'columns': List of string names for each column in the CSVs
    """
    if data_dir is None:
        # Resolve path relative to the installed package directory
        # package lives in src/mfml_qc/ so we go two levels up to find the root 'data/benzene'
        data_dir = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "data", "benzene")
        )

    xyz_path = os.path.join(data_dir, "traj.xyz")
    csv_path = os.path.join(data_dir, "energies.csv")
    time_path = os.path.join(data_dir, "timecosts.csv")
    cm_cache = os.path.join(data_dir, "X_CM.npy")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Benzene dataset not found at {data_dir}. "
            "Ensure you have placed the files in your project's 'data/benzene' directory. If you have not installed the dataset in the pip installation, please do so."
        )

    # Generate CM or load if already saved
    if os.path.exists(cm_cache):
        X_CM = np.load(cm_cache)
    else:
        X_CM = generate_coulomb_matrices(xyz_path, save_path=cm_cache)

    # Energies and Time costs
    energies = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    timecosts = np.genfromtxt(time_path, delimiter=",", skip_header=1)

    # Define exact columns representing the CSV structure
    columns = [
        "Time",
        "ZINDO",
        "LC-DFTB",
        "STO-3G",
        "3-21G",
        "6-31G",
        "def2-SVP",
        "def2-TZVP",
        "def2-QZVP",
    ]

    return {
        "X_CM": X_CM,
        "energies": energies,
        "timecosts": timecosts,
        "columns": columns,
    }
