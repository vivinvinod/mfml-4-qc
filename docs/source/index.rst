=====================================
Welcome to MFML-4-QC's documentation!
=====================================

**MFML-4-QC** is a Python package for Multifidelity Machine Learning in Quantum Chemistry. It provides an in built, Numba-accelerated Kernel Ridge Regression backend, dynamic active learning, and automated interfaces to quantum chemistry engines like ORCA.

.. toctree::
   :maxdepth: 2
   :caption: User Guide & Examples

   auto_examples/index

API Reference
=============

Representations
---------------
.. automodule:: mfml_qc.representations
   :members:
   :undoc-members:
   :show-inheritance:

Data Loaders & Utilities
------------------------
.. automodule:: mfml_qc.datasets
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfml_qc.utils
   :members:
   :undoc-members:
   :show-inheritance:

Kernel Ridge Regression
-------
.. automodule:: mfml_qc.krr
   :members:
   :undoc-members:
   :show-inheritance:

Quantum Chemistry Oracles
-------------------------
.. automodule:: mfml_qc.oracles.base
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfml_qc.oracles.orca
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: mfml_qc.oracles.pyscf
   :members:
   :undoc-members:
   :show-inheritance:


Multifidelity Machine Learning
---------
.. currentmodule:: mfml_qc

.. autoclass:: mfml_qc.mfml.ModelMFML
   :members:
   :undoc-members:
   :show-inheritance:

Active Learning
---------------
.. automodule:: mfml_qc.active_learning
   :members:
   :undoc-members:
   :show-inheritance:


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
