<div align="center">
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="mfml_logo_dark.png">
  <source media="(prefers-color-scheme: light)" srcset="mfml_logo_light.png">
  <p align="center">
  <img alt="MFML-4-QC logo" src="mfml_logo_light.png" width="400">
  </p>
</picture>
</div>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

MFML-4-QC is an open-source library that enables multifidelity machine learning for quantum chemical systems. While the multifidelity methods are model-architecture agnostic, this library provides a lightweight, ultra-fast Numba-compiled Kernel Ridge Regression (KRR) setup as the primary architecture. Users can seamlessly integrate their own custom ML models (e.g., from scikit-learn) and directly interface with quantum chemistry engines like ORCA and PySCF for automated data generation and active learning.

## Bug Reports  
If you find a bug in MFML-4-QC, or have a feature request, please open a GitHub issue.

## Key Features
* **Ultra-Fast Kernels:** Compute Matérn, Gaussian, Laplacian, and Wasserstein kernels efficiently using JIT-compiled C-loops via Numba.
* **In-Memory Representations:** Generate flattened Coulomb Matrices directly from .xyz trajectories without slow disk I/O.
* **Flexible ML Architectures:** Use the built-in KRR or drop in any scikit-learn compatible estimator (e.g., RandomForestRegressor, MLPRegressor).
* **Quantum Chemistry Oracles:** Abstract interfaces to automatically generate inputs, execute runs, and parse outputs from engines like ORCA and PySCF.

## Installation (v0.1.0 beta)
**Prerequisites:** Python 3.10 or higher. We highly recommend using a `conda` virtual environment to manage dependencies.

### Standard Installation
```bash
# Clone the repository
git clone https://github.com/vivinvinod/mfml-qc.git

# Create and activate a fresh conda environment
conda create -n mfmlenv python=3.10 -y
conda activate mfmlenv

# Install the package
pip install .
```


### Additional Dependencies
If you plan to use the built-in `PySCFEngine` oracle, you can install the package with the optional PySCF dependency. (Note: PySCF can be a heavy dependency, which is why it is kept optional).

```bash
pip install .[pyscf]
```

TO run the `ORCAEngine` you will need to install ORCA. See the [official ORCA manual](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/installation.html) for details on how to do so.

### Developer Installation
If you are beta testing, modifying the source code, or want to build the local documentation, install the package in "editable" mode (`-e`) with the `[dev]` flag. This installs testing tools (`pytest`, `black` etc) and the Sphinx documentation stack:

```bash
pip install -e .[dev]
```


## Documentation and Examples

MFML-4-QC comes with a comprehensive Sphinx-Gallery documentation site that includes detailed API references and copy-pasteable tutorials.

To view the documentation locally, go to `docs/build/html/` and open `index.html` in a browser of your choice. 

You can also browse the raw tutorial scripts directly in the `examples/` directory of this repository.


## Citation
If you use this package, please consider citing the following articles:
* TBA
* TBA





