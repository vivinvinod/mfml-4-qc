import os
import sys
# Tell Sphinx where the actual Python package lives
sys.path.insert(0, os.path.abspath('../../src'))

extensions = [
    'sphinx.ext.autodoc',      # Automatically pulls docstrings from classes/functions
    'sphinx.ext.napoleon',     # Allows Sphinx to read Google/NumPy style docstrings
    'sphinx.ext.viewcode',     # Adds a "[source]" button next to documentation to view the raw Python code
    'sphinx_gallery.gen_gallery', # Turns your examples/ folder into tutorials
    'nbsphinx',                # Renders Jupyter notebooks
]

# Configure Sphinx Gallery
sphinx_gallery_conf = {
    'examples_dirs': '../../examples',   # Path to your example scripts
    'gallery_dirs': 'auto_examples',     # Path where sphinx will save the generated HTML pages
}

# Set the theme to the scikit-learn style
html_theme = 'pydata_sphinx_theme'
