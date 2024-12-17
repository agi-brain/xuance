# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XuanCe'
copyright = '2023, XuanCe contributors'
author = 'XuanCe contributors'
release = 'v1.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # before that is renku, sphinx_rtd_theme
html_static_path = ['figures']
html_theme_options = {
    "light_logo": "logo_2.png",
    "dark_logo": "logo_2.png",
}

# The master toctree document.
master_doc = 'index'
