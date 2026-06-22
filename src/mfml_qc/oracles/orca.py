import numpy as np
import os
import subprocess
import glob
from typing import Dict, Any, Union
from .base import QuantumEngine

class OrcaEngine(QuantumEngine):
    """Concrete implementation for the ORCA quantum chemistry program."""
    
    def __init__(self, orca_path: str = "/bin/orca", properties_to_extract: Dict[str, str] = None):
        """
        Initializes the engine.
        
        Args:
            orca_path: Full path to the ORCA executable. Defaults to '/bin/orca' assuming it's in the PATH.
            properties_to_extract: Dictionary mapping property names to the exact string to search for in the output file.
        """
        self.orca_path = orca_path
        if properties_to_extract is None:
            # Default fallback behaviour
            self.properties_to_extract = {"energy": "FINAL SINGLE POINT ENERGY"}
        else:
            self.properties_to_extract = properties_to_extract

    def generate_input(self, geometry: Union[str, tuple], fidelity_params: Dict[str, Any], work_dir: str) -> str:
        os.makedirs(work_dir, exist_ok=True)
        input_path = os.path.join(work_dir, "calc.inp")
        
        charge = fidelity_params.get("charge", 0)
        multiplicity = fidelity_params.get("multiplicity", 1)
        template_file = fidelity_params.get("template_file", None)
        
        if template_file and os.path.exists(template_file):
            # Read from user-provided template file
            with open(template_file, 'r') as f:
                inp_content = f.read()
            # Ensure it ends with a newline before appending geometry
            if not inp_content.endswith("\n"):
                inp_content += "\n"
        else:
            # Dynamically build the input from scratch
            method = fidelity_params.get("method", "B3LYP")
            basis = fidelity_params.get("basis", "def2-SVP")
            
            # New parameters extracted from your shell script logic
            nprocs = fidelity_params.get("nprocs", 1)
            maxcore = fidelity_params.get("maxcore", None)
            optional_tags = fidelity_params.get("optional_tags", [])  # e.g., ["TightSCF", "RIJCOSX"]
            custom_blocks = fidelity_params.get("custom_blocks", "")  # e.g., "%tddft\n  nroots 10\nend"
            
            
            tags_str = " ".join(optional_tags)
            # EnGrad remains
            inp_content = f"! {method} {basis} EnGrad {tags_str}\n"
            
            
            if maxcore is not None:
                inp_content += f"%maxcore {maxcore}\n"
            if nprocs > 1:
                inp_content += f"%PAL nproc {nprocs} end\n"
            if custom_blocks:
                inp_content += f"{custom_blocks}\n"
                
            inp_content += "\n"

        if isinstance(geometry, str):
            geometry_path = os.path.abspath(geometry)
        else:
            geometry_path = geometry
            
        inp_content += f"* xyzfile {charge} {multiplicity} {geometry_path}\n" 
        
        with open(input_path, 'w') as f:
            f.write(inp_content)
            
        return input_path

    def run_calculation(self, input_file: str, work_dir: str) -> str:
        output_file = input_file.replace(".inp", ".out")

        work_dir_abs = os.path.abspath(work_dir)
        input_file_abs = os.path.abspath(input_file)
        output_file_abs = os.path.abspath(output_file)
        
        # Run ORCA using subprocess
        # `orca calc.inp > calc.out`
        with open(output_file, 'w') as out_f:
            try:
                subprocess.run(
                    [self.orca_path, input_file_abs],
                    stdout=out_f,
                    stderr=subprocess.STDOUT,
                    cwd=work_dir_abs,
                    check=True # Raises exception if ORCA crashes
                )
            except subprocess.CalledProcessError:
                pass 
                
        return output_file

    def parse_output(self, output_file: str, parse_gradients=False, parse_spectra=False) -> Dict[str, Any]:
        results = {"success": False}
        for prop in self.properties_to_extract.keys():
            results[prop] = None
        
        if not os.path.exists(output_file):
            return results

        with open(output_file, 'r') as f:
            lines = f.readlines()
            
        # check if ORCA terminated normally
        if any("ORCA TERMINATED NORMALLY" in line for line in lines[-10:]):
            results["success"] = True
            
        #  Extract Property
        for line in reversed(lines):
            for prop_name, search_string in self.properties_to_extract.items():
                if results[prop_name] is None and search_string in line:
                    # Extract the last valid number on the line (ignores units like 'Eh' or trailing labels)
                    words = line.split()
                    for word in reversed(words):
                        try:
                            results[prop_name] = float(word)
                            break # Found the float!
                        except ValueError:
                            pass
        
        #gradients if engrad
        if parse_gradients:
            engrad_file = output_file.replace(".out", ".engrad")
            if os.path.exists(engrad_file):
                with open(engrad_file, 'r') as f:
                    eg_lines = f.readlines()
                
                n_atoms = 0
                gradients = []
                for i, line in enumerate(eg_lines):
                    if "Number of atoms" in line:
                        n_atoms = int(eg_lines[i+2].strip())
                    elif "The current gradient" in line:
                        start_idx = i + 2
                        for j in range(n_atoms * 3):
                            gradients.append(float(eg_lines[start_idx + j].strip()))
                        break
                        
                if gradients and n_atoms > 0:
                    results["gradients"] = np.array(gradients).reshape(n_atoms, 3)
        
        #extract spectra details
        if parse_spectra:
            spectra = []
            for i, line in enumerate(lines):
                if "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in line or \
                   "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in line:
                    # Table data typically starts 5 lines after the header
                    start_idx = i + 5
                    for j in range(start_idx, len(lines)):
                        if "--------" in lines[j] or lines[j].strip() == "":
                            break # End of table
                        parts = lines[j].split()
                        if len(parts) >= 8:
                            spectra.append({
                                "state": int(parts[0]),
                                "energy_cm1": float(parts[1]),
                                "wavelength_nm": float(parts[2]),
                                "fosc": float(parts[3]),
                                "px": float(parts[5]), # Index 4 is P2, 5 is PX
                                "py": float(parts[6]),
                                "pz": float(parts[7])
                            })
                    # Save it and break to avoid parsing earlier identical blocks
                    results["tddft_spectrum"] = spectra
                    break

        return results

    def cleanup(self, work_dir: str):
        """Deletes large cache files generated by ORCA."""
        patterns_to_delete = ["*.tmp", "*.dens", "*.gbw", "*.prop"]
        for pattern in patterns_to_delete:
            for f in glob.glob(os.path.join(work_dir, pattern)):
                try:
                    os.remove(f)
                except OSError:
                    pass
                    
                    
'''
engine = OrcaEngine(
    orca_path="/path/to/orca", 
    properties_to_extract={
        "energy": "FINAL SINGLE POINT ENERGY",
        "homo_lumo_gap": "HOMO-LUMO GAP"
    } #The parser will loop through the file from bottom to top, find the target string, and attempt to grab the numerical value at the end of the line. 
)

#provide self input file:
fidelity_settings = {
    "template_file": "examples/example_input.inp", 
    "charge": 0, 
    "multiplicity": 1
}
#or provide fidelity details
#fidelity_settings = {
#    "method": "CAM-B3LYP",
#    "basis": "3-21G",
#    "optional_tags": ["def2/J", "TightSCF", "RIJCOSX"],
#    "nprocs": 64,
#    "maxcore": 2000,
#    "custom_blocks": "%tddft\n   ETol 1e-6\n   RTol 1e-6\n   nroots 10\n   maxdim 100\n   tprint 1E-10\n   triplets false\nend"
#}

results = engine.evaluate(geometry="/path/to/molecule_1.xyz", 
                          fidelity_params=fidelity_settings, 
                          work_dir="tmp/")

'''

