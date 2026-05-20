# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

project = "Breda Guardians Esports Tool"
copyright = "2026, Endijs Kirsteins, Rachit Ravelia, Raf Sikkema, Kamil Lega, Kees Klijs, Louie Dans, Jack Wade, Tiago Alberto Silva"
author = "Endijs Kirsteins, Rachit Ravelia, Raf Sikkema, Kamil Lega, Kees Klijs, Louie Dans, Jack Wade, Tiago Alberto Silva"
release = "0.0.1"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Enable useful extensions: autodoc (for automatic docs), viewcode (link to source),
# and napoleon (Google/NumPy style docstrings). Autodoc requires the project
# modules to be importable from the docs build; we add the backend path below.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

# Ensure the backend package (backend/) is importable as a top-level path so
# autodoc can import modules like ``app.router.feedback``. The backend folder
# lives two levels up from this conf.py (repo_root/backend).
HERE = os.path.abspath(os.path.dirname(__file__))
BACKEND_PATH = os.path.abspath(os.path.join(HERE, "..", "..", "backend"))
if BACKEND_PATH not in sys.path:
    sys.path.insert(0, BACKEND_PATH)

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Sometimes the backend imports optional typing helpers not present in the
# docs build environment (for example `typing_extensions`). Mock them so
# autodoc can import the modules without installing every dev dependency.
autodoc_mock_imports = ["typing_extensions", "fastapi", "pydantic", "sqlalchemy"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]
