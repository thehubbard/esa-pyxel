#
# pyxel documentation build configuration file, created by
# sphinx-quickstart on Tue Nov 21 09:37:42 2017.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

from datetime import datetime
from pathlib import Path

from setuptools.config import read_configuration

import pyxel

# Read 'setup.cfg' file
parent_folder = Path(__file__).parent
setup_cfg_filename = parent_folder.joinpath("../../setup.cfg").resolve(
    strict=True
)  # type: Path
metadata = read_configuration(setup_cfg_filename)["metadata"]  # type: dict

now_dt = datetime.now()  # type: datetime

# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",  # include documentation from docstrings
    "sphinx.ext.autosummary",  # Generate autodoc summaries
    "sphinx.ext.extlinks",
    "sphinx.ext.coverage",  # collect doc coverage stats
    "sphinx.ext.mathjax",  # render math via Javascript
    "sphinx.ext.napoleon",  # for numpy docstyle
    "sphinx.ext.todo",  # support for todo items (.. todo::)
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.viewcode",  # add links to highlighted source code
    "sphinxcontrib.bibtex",
    "sphinx_panels",
    "sphinx_inline_tabs",  # Add inline tabs
    "sphinx.ext.intersphinx",  # Link to other project's documentation
]

extlinks = {
    "issue": ("https://gitlab.com/esa/pyxel/-/issues/%s", "GH"),
    "pull": ("https://gitlab.com/esa/pyxel/-/merge_requests/%s", "PR"),
}
bibtex_bibfiles = ["refs.bib"]


autodoc_typehints = "none"

# Napoleon configurations
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_rtype = False
napoleon_preprocess_types = True
napoleon_type_aliases = {
    # Objects related to a detector
    "Detector": "~pyxel.detectors.Detector",
    "CCD": "~pyxel.detectors.CCD",
    "CMOS": "~pyxel.detectors.CMOS",
    "MKID": "~pyxel.detectors.MKID",
    # Objects related to a running mode
    "Exposure": "~pyxel.exposure.Exposure",
    "Observation": "~pyxel.observation.Observation",
    "ObservationResult": "~pyxel.observation.ObservationResult",
    "Calibration": "~pyxel.calibration.Calibration",
    # Objects related to a pipeline
    "DetectionPipeline": "~pyxel.pipelines.DetectionPipeline",
    # General terms,
    "Sequence": ":term:`sequence`",
    "Path": ":py:class:`Path <pathlib.Path>`",
    # XArray
    "Dataset": "~xarray.Dataset",
    # Pandas
    "DataFrame": "~pandas.DataFrame",
    # Numpy
    "ndarray": "~numpy.ndarray",
    # Holoviews
    "hv.Points": "~holoviews.element.Points",
    "hv.Layout": "~holoviews.core.Layout",
}

autodoc_member_order = "bysource"

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = [".rst", ".md"]
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# General information about the project.
project = metadata["name"]
copyright = f"2017-{now_dt:%Y}, European Space Agency"
author = "Pyxel Developers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = pyxel.__version__
# The full version, including alpha/beta/rc tags.
release = pyxel.__version__
tag = version.split("+")[0]

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
today_fmt = "%Y-%m-%d"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpointƒs"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_book_theme"
html_title = f"version {version}"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "repository_url": "https://gitlab.com/esa/pyxel",
    # Add buttons
    "use_edit_page_button": True,
    "use_repository_button": True,
    "use_issues_button": True,
    "use_download_button": True,
    "home_page_in_toc": False,
    "extra_navbar": "",
    "navbar_footer_text": "",
    "extra_footer": f"""<p>Last updated on {now_dt:%Y-%m-%d}.</p>""",
    "toc_title": "Contents",  # Control the right sidebar items
}

# This is used to generate the link 'suggest edit'
html_context = {
    "github_url": "https://gitlab.com",
    "github_user": "esa",
    "github_repo": "pyxel",
}
html_theme_options["path_to_docs"] = "../../-/edit/master/docs/source"


# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "_static/pyxel-logo.png"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
html_css_files = ["style.css"]

# configuration for sphinxext.opengraph
ogp_site_url = "https://esa.gitlab.io/pyxel/doc/"
ogp_image = "https://esa.gitlab.io/pyxel/doc/_static/esa-logo.png"
ogp_custom_meta_tags = [
    # '<meta name="twitter:card" content="summary_large_image" />',
    '<meta name="image" property="og:image" content="https://esa.gitlab.io/pyxel/doc/_static/esa-logo.png">',
]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# This is required for the alabaster theme
# refs: http://alabaster.readthedocs.io/en/latest/installation.html#sidebars
# html_sidebars = {
#     "**": [
#         "relations.html",  # needs 'show_related': True theme option to display
#         "searchbox.html",
#     ]
# }


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = metadata["name"] + "doc"


# https://xarray.pydata.org/en/stable/objects.inv
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    # "iris": ("https://scitools-iris.readthedocs.io/en/latest", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    # "scipy": ("https://docs.scipy.org/doc/scipy", None),
    # "numba": ("https://numba.pydata.org/numba-doc/latest", None),
    # "matplotlib": ("https://matplotlib.org/stable/", None),
    "dask": ("https://docs.dask.org/en/latest", None),
    # "cftime": ("https://unidata.github.io/cftime", None),
    # "rasterio": ("https://rasterio.readthedocs.io/en/latest", None),
    "xarray": ("https://xarray.pydata.org/en/stable/", None),
    "holoviews": ("https://holoviews.org/", None),
}


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (
        master_doc,
        metadata["name"] + ".tex",
        metadata["name"] + " Documentation",
        metadata["author"],
        "manual",
    )
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, metadata["name"], metadata["name"] + " Documentation", [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        metadata["name"],
        metadata["name"] + " Documentation",
        author,
        metadata["name"],
        "One line description of project.",
        "Miscellaneous",
    )
]


def html_page_context(app, pagename, templatename, context, doctree):
    # Disable edit button for docstring generated pages
    if "generated" in pagename:
        context["theme_use_edit_page_button"] = False


def setup(app):
    app.connect("html-page-context", html_page_context)
