from abc import ABC, abstractmethod
from typing import Dict, Any, Union
import numpy as np

class QuantumEngine(ABC):
    """
    Abstract interface for QC engines.
    Defines the lifecycle of a single calculation.
    """
    def evaluate(self, geometry: Union[str, tuple], fidelity_params: Dict[str, Any], work_dir: str) -> Dict[str, Any]:
        """
        The main orchestrator method called by the Active Learning loop.
        Handles the entire lifecycle of the calculation.
        """
        input_file = self.generate_input(geometry, fidelity_params, work_dir)
        output_file = self.run_calculation(input_file, work_dir)
        results = self.parse_output(output_file)
        self.cleanup(work_dir)
        
        return results

    @abstractmethod
    def generate_input(self, geometry: Union[str, tuple], fidelity_params: Dict[str, Any], work_dir: str) -> str:
        """Generates the software-specific input file. Returns the path to the input file."""
        pass

    @abstractmethod
    def run_calculation(self, input_file: str, work_dir: str) -> str:
        """Executes the quantum chemistry program. Returns the path to the output file."""
        pass

    @abstractmethod
    def parse_output(self, output_file: str) -> Dict[str, Any]:
        """
        Parses the output file. 
        MUST return a dictionary containing at least a 'success': bool key.
        """
        pass

    @abstractmethod
    def cleanup(self, work_dir: str):
        """Removes temporary/scratch files to save disk space."""
        pass
