# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'autodiff'
copyright = '2024, Yuiko Ichikawa'
author = 'Yuiko Ichikawa'


import toml
# Read version from pyproject.toml
with open('../pyproject.toml', 'r') as f:
    pyproject = toml.load(f)
release = pyproject['project']['version'] # for the unified version control

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'nbsphinx', 'myst_parser']

templates_path = ['_templates']
exclude_patterns = ['.egg-info']

language = 'python3'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
