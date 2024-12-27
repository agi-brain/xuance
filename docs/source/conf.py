# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
print("[DOCS] xuance library path: {}".format(sys.path[0]))

project = 'XuanCe'
copyright = '2023, XuanCe Contributors.'
author = 'Wenzhang Liu, etc.'
release = "1.2"

# The master toctree document.
master_doc = 'index'

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
    # External stuff
    "myst_parser",
    "sphinx_copybutton",
    'sphinx_tabs.tabs',
    "sphinx_design",
    "sphinx_favicon",
    "notfound.extension",
    "sphinx_github_changelog"
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "dollarmath",  # Enables $...$ for inline and $$...$$ for block math
]

autodoc_mock_imports = [
    "numpy",
    "scipy",
    "gym",
    "gymnasium",
    "gym-notices",
    "pygame",
    "tqdm",
    "pyglet",
    "pettingzoo",  # for MARL
    "tensorboard",  # logger
    "wandb",
    "moviepy",
    "imageio",  # default version is 2.9.0
    "mpi4py",  # default version is 3.1.3
    "torch",
    "tensorflow",
    "tensorflow_probability",
    "tensorflow-addons",
    "mindspore",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # sphinx_rtd_theme (before that is renku)
html_title = f"<div style='text-align: center; font-size: 20px'><strong>{project}</strong></div>"
html_static_path = ['_static']
html_theme_options = {
    # logo
    "light_logo": "figures/logo_2.png",
    "dark_logo": "figures/logo_2.png",
    #
    "source_repository": "https://github.com/agi-brain/xuance",
    "source_branch": "../tree/master",
    "source_directory": "docs/source",
    "top_of_page_buttons": ["view", "edit"],
}
html_css_files = [
    'css/xuance.css',  # Name of xuance CSS file
]

favicons = [
    {"href": "figures/favicon/favicon.svg"},  # => use `_static/icon.svg`
    {"href": "figures/favicon/favicon-96x96.png"},
    {
        "rel": "apple-touch-icon",
        "href": "figures/favicon/apple-touch-icon.png",
    },
]

# -- Generate Changelog -------------------------------------------------

sphinx_github_changelog_token = os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN")
