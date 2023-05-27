# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XuanPolicy'
copyright = '2023, Wenzhang Liu, Wenzhe Cai, Kun Jiang, Yuanda Wang, Guangran Cheng, Jiawei Wang, Jingyu Cao, Lele Xu, Chaoxu Mu, Changyin Sun'
author = 'Wenzhang Liu, Wenzhe Cai, Kun Jiang, Yuanda Wang, Guangran Cheng, Jiawei Wang, Jingyu Cao, Lele Xu, Chaoxu Mu, Changyin Sun'
release = 'v0.1.7'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
