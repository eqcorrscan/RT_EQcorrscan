# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import glob

cwd = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(cwd)

sys.path.insert(0, project_root)

import rt_eqcorrscan

READ_THE_DOCS = os.environ.get('READTHEDOCS', None) == 'True'
if not READ_THE_DOCS:
    print('Imported rtd')
    import sphinx_rtd_theme
# -- Project information -----------------------------------------------------

project = 'RT-EQcorrscan'
copyright = '2019, Calum J. Chamberlain'
author = 'Calum J. Chamberlain'

# The full version, including alpha/beta/rc tags
# The short X.Y version.
version = rt_eqcorrscan.__version__
# The full version, including alpha/beta/rc tags.
release = rt_eqcorrscan.__version__



# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "numpydoc",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
if not READ_THE_DOCS:
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3.7/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'obspy': ('https://docs.obspy.org/', None),
    'obsplus': ('https://niosh-mining.github.io/obsplus/', None),
}

autosummary_generate = glob.glob("modules" + os.sep + "*.rst")

autoclass_content = 'class'

autodoc_default_flags = ['show-inheritance']

# warn about *all* references where the target cannot be found
nitpicky = False

trim_doctest_flags = True
