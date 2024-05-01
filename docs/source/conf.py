# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XuanCe'
copyright = '2023, XuanCe contributors'
author = 'XuanCe contributors'
release = 'v1.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'renku_sphinx_theme',
    'sphinx_copybutton',
    'sphinx_tabs.tabs',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# import sphinx_rtd_theme
# html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme = "renku"
import renku_sphinx_theme
html_theme_path = [renku_sphinx_theme.get_path()]
html_logo = "figures/logo_2.png"


# The master toctree document.
master_doc = 'index'
