import pytest
import os
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from mfml_qc.representations import (
    compute_flat_coulomb,
    parse_trajectory,
    generate_coulomb_matrices
)

############TESTS############

@pytest.fixture
def sample_xyz_file(tmp_path):
    """
    Creates a temporary valid XYZ file containing two water molecules.
    tmp_path is a built-in pytest fixture that provides a unique temporary directory.
    """
    content = """3
Water molecule 1
O 0.000 0.000 0.000
H 0.000 0.757 0.587
H 0.000 -0.757 0.587
3
Water molecule 2
O 0.000 0.000 0.000
H 0.000 0.760 0.590
H 0.000 -0.760 0.590
"""
    filepath = tmp_path / "test_water.xyz"
    filepath.write_text(content)
    return str(filepath)

@pytest.fixture
def mixed_atoms_xyz_file(tmp_path):
    """Creates an XYZ file where geometries have different numbers of atoms."""
    content = """2
Hydrogen
H 0.0 0.0 0.0
H 0.0 0.0 0.74
1
Hydrogen atom
H 0.0 0.0 0.0
"""
    filepath = tmp_path / "test_mixed.xyz"
    filepath.write_text(content)
    return str(filepath)


def test_compute_flat_coulomb():
    """
    Manually verifies the math of the Coulomb Matrix for an H2 molecule.
    """
    # H2 molecule, distance = 1.0 Angstrom
    Z = np.array([1, 1], dtype=np.int32)
    R = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    
    # Expected calculations:
    # C_11 = 0.5 * (1 ** 2.4) = 0.5
    # C_12 = (1 * 1) / 1.0 = 1.0
    # C_22 = 0.5 * (1 ** 2.4) = 0.5
    # Flat expected array: [C_11, C_12, C_22] -> [0.5, 1.0, 0.5]
    expected_c_flat = np.array([0.5, 1.0, 0.5])
    
    c_flat = compute_flat_coulomb(Z, R)
    
    assert c_flat.shape == (3,)
    assert_allclose(c_flat, expected_c_flat, rtol=1e-7)

def test_parse_trajectory(sample_xyz_file):
    """Tests the in-memory parsing of the concatenated XYZ file."""
    geometries = parse_trajectory(sample_xyz_file)
    
    # Should find 2 molecules
    assert len(geometries) == 2
    
    Z1, R1 = geometries[0]
    
    # Check that O, H, H mapped correctly to atomic numbers 8, 1, 1
    assert_array_equal(Z1, [8, 1, 1])
    
    # Check coordinate shapes and specific parsing
    assert R1.shape == (3, 3)
    assert R1[1, 1] == 0.757

def test_generate_coulomb_matrices(sample_xyz_file, tmp_path):
    """Tests the full CM generation pipeline and saving."""
    save_file = tmp_path / "water_cm.npy"
    
    X_CM = generate_coulomb_matrices(sample_xyz_file, save_path=str(save_file))
    
    # For a 3 atom molecule, num features = 3*(3+1)/2 = 6
    # 2 molecules in the sample file -> shape should be (2, 6)
    assert X_CM.shape == (2, 6)
    
    # Verify the save path actually created the file
    assert os.path.exists(save_file)
    
    # Verify the loaded data matches the returned data
    loaded_CM = np.load(save_file)
    assert_allclose(X_CM, loaded_CM)

def test_generate_coulomb_matrices_exceptions(tmp_path, mixed_atoms_xyz_file):
    """Verifies that the function safely catches and handles bad input."""
    
    # 1. File not found
    with pytest.raises(FileNotFoundError):
        generate_coulomb_matrices("does_not_exist_at_all.xyz")
        
    # 2. Empty file
    empty_file = tmp_path / "empty.xyz"
    empty_file.write_text("")
    with pytest.raises(ValueError, match="appears to be empty"):
        generate_coulomb_matrices(str(empty_file))
        
    # 3. Inconsistent atom counts
    with pytest.raises(ValueError, match="require consistent atom counts"):
        generate_coulomb_matrices(mixed_atoms_xyz_file)
