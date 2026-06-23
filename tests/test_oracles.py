import pytest
import os
import json
from mfml_qc.oracles import OrcaEngine, PySCFEngine

######ORCA TESTS


def test_orca_generate_input(tmp_path):
    """Tests if ORCA input files are formatted correctly without running ORCA."""
    engine = OrcaEngine()
    work_dir = str(tmp_path)
    work_dir = os.path.abspath(work_dir)

    fidelity_params = {
        "method": "PBE",
        "basis": "cc-pVDZ",
        "charge": 1,
        "multiplicity": 2,
        "nprocs": 4,
        "maxcore": 1000,
        "EnGrad": "EnGrad",
        "optional_tags": ["TightSCF"],
    }

    input_file = engine.generate_input("molecule.xyz", fidelity_params, work_dir)
    input_file = os.path.abspath(input_file)

    assert os.path.exists(input_file)
    with open(input_file, "r") as f:
        content = f.read()

    # Verify exact string injections
    assert "! PBE cc-pVDZ EnGrad TightSCF" in content
    assert "%maxcore 1000" in content
    assert "%PAL nproc 4 end" in content
    assert "* xyzfile 1 2" in content


def test_orca_parse_output_success(tmp_path):
    """Tests if the ORCA parser correctly extracts energy from a successful run."""
    engine = OrcaEngine()
    dummy_out = tmp_path / "calc.out"

    # Create a fake ORCA output file
    content = """
... lots of junk text ...
-------------------------   --------------------
FINAL SINGLE POINT ENERGY       -76.432198765432
-------------------------   --------------------
... more junk ...
****ORCA TERMINATED NORMALLY****
"""
    dummy_out.write_text(content)

    results = engine.parse_output(str(dummy_out))

    assert results["success"] is True
    assert results["energy"] == -76.432198765432


def test_orca_parse_output_failure(tmp_path):
    """Tests if the ORCA parser safely handles a crashed run."""
    engine = OrcaEngine()
    dummy_out = tmp_path / "calc.out"

    # Missing the "TERMINATED NORMALLY" string
    content = "SCF DID NOT CONVERGE\nFINAL SINGLE POINT ENERGY -76.000\n"
    dummy_out.write_text(content)

    results = engine.parse_output(str(dummy_out))

    assert results["success"] is False


############PYSCF tests


def test_pyscf_generate_input(tmp_path):
    """Tests if PySCF generates the correct standalone Python script."""
    engine = PySCFEngine()
    work_dir = str(tmp_path)

    fidelity_params = {
        "method": "B3LYP",
        "basis": "def2-SVP",
        "charge": 0,
        "multiplicity": 1,  # spin = 0
    }

    input_file = engine.generate_input("data.xyz", fidelity_params, work_dir)

    assert os.path.exists(input_file)
    with open(input_file, "r") as f:
        content = f.read()

    assert "mol = gto.M(" in content
    assert "spin=0" in content
    assert "mf.xc = 'B3LYP'" in content
    assert "json.dump(results, f)" in content


def test_pyscf_parse_output(tmp_path):
    """Tests if PySCF parses the dumped JSON correctly."""
    engine = PySCFEngine()
    dummy_json = tmp_path / "pyscf_results.json"

    # Write fake successful JSON
    dummy_data = {"success": True, "energy": -100.1234}
    dummy_json.write_text(json.dumps(dummy_data))

    results = engine.parse_output(str(dummy_json))

    assert results["success"] is True
    assert results["energy"] == -100.1234
