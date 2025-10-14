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
release = "1.3.2"

# The master toctree document.
master_doc = 'index'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # Sphinx's own extensions
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
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

napoleon_google_docstring = True
napoleon_numpy_docstring = False

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
    "gymnasium",
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
    "optuna",
    "optuna-dashboard",
    "plotly",
]

pygments_style = "tango"
pygments_dark_style = "zenburn"

intersphinx_disabled_domains = ["std"]
templates_path = ['_templates']
exclude_patterns = []
rst_prolog = """
.. include:: <s5defs.txt>

.. |_1| unicode:: 0xA0
    :trim:

.. |_2| unicode:: 0xA0 0xA0
    :trim:

.. |_3| unicode:: 0xA0 0xA0 0xA0
    :trim:

.. |_4| unicode:: 0xA0 0xA0 0xA0 0xA0
    :trim:

.. |_5| unicode:: 0xA0 0xA0 0xA0 0xA0 0xA0
    :trim:

.. |torch| image:: /_static/figures/DL_tools_logo/pytorch.svg
    :width: 18
    :align: middle

.. |tensorflow| image:: /_static/figures/DL_tools_logo/tensorflow.svg
    :width: 20
    :align: middle

.. |mindspore| image:: /_static/figures/DL_tools_logo/mindspore.svg
    :width: 36
    :align: middle
"""

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"  # sphinx_rtd_theme (before that is renku)
html_title = f"<div style='text-align: center; font-size: 20px'><strong>{project}</strong></div>"
html_short_title = "XuanCe"
html_scaled_image_link = False
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
    # color
    # "light_css_variables": {
    #     "color-brand-primary": "#7C4DFF",
    #     "color-brand-content": "#7C4DFF",
    # },
    "navigation_with_keys": True,  # Controls whether the user can navigate the documentation using the keyboard’s left and right arrows.
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

sphinx_github_changelog_token = (
    os.environ.get("SPHINX_GITHUB_CHANGELOG_TOKEN") or
    os.environ.get("GITHUB_TOKEN")
)
