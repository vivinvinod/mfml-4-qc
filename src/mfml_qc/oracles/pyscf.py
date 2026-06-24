import os
import sys
import json
import subprocess
from typing import Dict, Any, Union
from .base import QuantumEngine


class PySCFEngine(QuantumEngine):
    """
    Implementation for the PySCF library as the quantum chemsitry engine.
    This script is untested and is under works.
    """

    def __init__(self, python_executable: str = sys.executable):
        """
        Initializes the PySCF engine.

        Args:
            python_executable: Path to the Python executable that has PySCF installed.
                               Defaults to the environment currently running the script.
        """
        self.python_executable = python_executable

    def generate_input(
        self,
        geometry: Union[str, tuple],
        fidelity_params: Dict[str, Any],
        work_dir: str,
    ) -> str:
        os.makedirs(work_dir, exist_ok=True)
        input_path = os.path.join(work_dir, "run_pyscf.py")

        method = fidelity_params.get("method", "B3LYP")
        basis = fidelity_params.get("basis", "def2-SVP")
        charge = fidelity_params.get("charge", 0)

        # PySCF uses "spin" (number of unpaired electrons, 2S)
        # ORCA uses "multiplicity" (2S + 1)
        multiplicity = fidelity_params.get("multiplicity", 1)
        spin = multiplicity - 1

        # script that executes PySCF and dumps results to JSON.
        # Prevents PySCF memory leaks
        script_content = f"""
import json
from pyscf import gto, scf, dft

try:
    # extract geomtry
    mol = gto.M(
        atom='{geometry}',
        basis='{basis}',
        charge={charge},
        spin={spin},
        verbose=3 # Suppress excessive printing
    )

    # method 
    method_name = '{method}'.upper()
    if method_name == 'HF':
        mf = scf.RHF(mol) if {spin} == 0 else scf.UHF(mol)
    else:
        mf = dft.RKS(mol) if {spin} == 0 else dft.UKS(mol)
        mf.xc = '{method}'
        
    # mf.conv_tol = 1e-8 

    # Run the calculation
    energy = mf.kernel()
    success = mf.converged

    # Save results
    results = {{
        "success": success,
        "energy": energy
    }}
    
except Exception as e:
    results = {{
        "success": False,
        "error": str(e)
    }}

with open("pyscf_results.json", "w") as f:
    json.dump(results, f)
"""
        with open(input_path, "w") as f:
            f.write(script_content.strip())

        return input_path

    def run_calculation(self, input_file: str, work_dir: str) -> str:
        # PySCF output will be dumped to this JSON file by the above generated script
        output_file = os.path.join(work_dir, "pyscf_results.json")

        try:
            subprocess.run(
                [self.python_executable, input_file],
                cwd=work_dir,
                check=True,
                capture_output=True,  # Capture stdout so it doesn't spam the main run (say in AL)
            )
        except subprocess.CalledProcessError:
            pass

        return output_file

    def parse_output(self, output_file: str) -> Dict[str, Any]:
        results = {"success": False, "energy": None}

        if not os.path.exists(output_file):
            return results

        with open(output_file, "r") as f:
            try:
                parsed_data = json.load(f)
                results.update(parsed_data)
            except json.JSONDecodeError:
                pass

        return results

    def cleanup(self, work_dir: str):
        """Removes PySCF generated scripts and temp files."""
        files_to_delete = [
            "run_pyscf.py",
            "pyscf_results.json",
            "chkfile.chk",  # PySCF default checkpoint file name
        ]

        for f in files_to_delete:
            full_path = os.path.join(work_dir, f)
            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass
