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
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    # Our custom extension, only meant for Furo's own documentation.
    "furo.sphinxext",
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    'sphinx_tabs.tabs',
    "sphinx_design",
    "sphinx_inline_tabs",
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

html_favicon = 'figures/favicon.jpeg'

# The master toctree document.
master_doc = 'index'
