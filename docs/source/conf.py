# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the root directory of the project to the Python path
sys.path.insert(0, os.path.abspath('../../'))

# Ensure the Python path includes the `silverspeak` directory
sys.path.insert(0, os.path.abspath('../../silverspeak'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SilverSpeak'
copyright = '2025, Aldan Creo'
author = 'Aldan Creo'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme'
]

templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Add custom CSS file
html_css_files = ['custom.css']

# Logo configuration
html_logo = '_static/silverspeak_logo_editable.svg'
html_theme_options = {
    'logo_only': False,
    'display_version': True,
}

# Favicon
html_favicon = '_static/silverspeak_logo_editable.svg'
