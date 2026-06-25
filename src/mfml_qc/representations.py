import os
import numpy as np
from numba import njit, prange
from tqdm.auto import tqdm

ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}


@njit(fastmath=True)
def compute_flat_coulomb(Z: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    Computes the 1D flattened unsorted Coulomb matrix (diagonal + upper triangle)
    using ultra-fast compiled C loops.

    Parameters
    ----------
    Z : np.ndarray
        1D array of nuclear charges (shape: N,)
    R : np.ndarray
        2D array of Cartesian coordinates (shape: N, 3)

    Returns
    -------
    np.ndarray
        1D array of flattened Coulomb matrix features.
    """
    N = Z.shape[0]
    num_features = int((N * (N + 1)) / 2)
    C_flat = np.zeros((num_features), dtype=np.float64)

    idx = 0
    for i in prange(N):
        # Diagonal element (approximate polynomial fit of atomic energies)
        C_flat[idx] = 0.5 * (Z[i] ** 2.4)
        idx += 1

        # Upper triangular elements (Coulomb repulsion)
        for j in range(i + 1, N):
            # Manual distance calculation (highly optimized for Numba)
            dx = R[i, 0] - R[j, 0]
            dy = R[i, 1] - R[j, 1]
            dz = R[i, 2] - R[j, 2]
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            # Coulomb term
            C_flat[idx] = (Z[i] * Z[j]) / dist
            idx += 1

    return C_flat


def parse_trajectory(filepath: str) -> list:
    """
    Reads a concatenated XYZ file entirely in memory, avoiding slow disk I/O.

    Parameters
    ----------
    filepath : str
        Path to the concatenated .xyz file.

    Returns
    -------
    list of tuples
        A list where each element is a tuple of (nuclear_charges, coordinates).
    """
    geometries = []

    with open(filepath, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break  # End of file

            line = line.strip()
            if not line:
                continue  # Skip empty lines

            n_atoms = int(line)
            comment = f.readline().strip()  # Skip comment line

            Z = np.zeros(n_atoms, dtype=np.int32)
            R = np.zeros((n_atoms, 3), dtype=np.float64)

            for i in range(n_atoms):
                parts = f.readline().split()
                Z[i] = ATOMIC_NUMBERS[parts[0].capitalize()]
                R[i, 0] = float(parts[1])
                R[i, 1] = float(parts[2])
                R[i, 2] = float(parts[3])

            geometries.append((Z, R))

    return geometries


def generate_coulomb_matrices(xyz_filepath: str, save_path: str = None) -> np.ndarray:
    r"""
    Extracts geometries from a concatenated XYZ file and generates
    flattened, unsorted Coulomb matrices for the entire dataset.

    The Coulomb matrix :math:`C` is a global structural representation defined as:
    
    .. math::
        C_{ij} = \begin{cases}
        0.5 Z_i^{2.4} & \text{for } i = j \\
        \frac{Z_i Z_j}{||\mathbf{R}_i - \mathbf{R}_j||_2} & \text{for } i \neq j
        \end{cases}
           
    where :math:`Z_i` is the atomic number and :math:`\mathbf{R}_i` is the Cartesian coordinate of atom :math:`i`.
    This function flattens the upper triangle (including the diagonal) into a 1D vector.

    Parameters
    ----------
    xyz_filepath : str
        Path to the source .xyz file.
    save_path : str, optional
        If provided, saves the output array to this filepath as a .npy file
        (e.g., 'data/CH3Cl_CM.npy').

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_geometries, n_features) containing
        the flattened Coulomb matrices.

    Raises
    ------
    FileNotFoundError
        If the specified XYZ file does not exist.
    ValueError
        If the XYZ file is empty, or if geometries within the trajectory
        have inconsistent numbers of atoms.
    """
    if not os.path.exists(xyz_filepath):
        raise FileNotFoundError(f"Could not find the dataset at {xyz_filepath}")

    # Read all geometries in memory
    geometries = parse_trajectory(xyz_filepath)
    n_samples = len(geometries)

    if n_samples == 0:
        raise ValueError("The provided XYZ file appears to be empty.")

    # Determine feature size based on the first molecule
    n_atoms = geometries[0][0].shape[0]
    n_features = int((n_atoms * (n_atoms + 1)) / 2)

    # Initialize the final feature matrix
    X_CM = np.zeros((n_samples, n_features), dtype=np.float64)

    # Generate Coulomb matrices
    for i in tqdm(range(n_samples), desc="Generating Unsorted CMs", leave=True):
        Z_i, R_i = geometries[i]

        if Z_i.shape[0] != n_atoms:
            raise ValueError(
                f"Geometry {i} has {Z_i.shape[0]} atoms, expected {n_atoms}. "
                "Coulomb matrices require consistent atom counts."
            )

        X_CM[i, :] = compute_flat_coulomb(Z_i, R_i)

    # Optionally save to disk
    if save_path:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.save(save_path, X_CM)

    return X_CM
