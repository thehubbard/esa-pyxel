#  Copyright (c) European Space Agency, 2017, 2018, 2019, 2020.
#
#  This file is subject to the terms and conditions defined in file 'LICENCE.txt', which
#  is part of this Pyxel package. No part of the package, including
#  this file, may be copied, modified, propagated, or distributed except according to
#  the terms contained in the file ‘LICENCE.txt’.
import importlib
import locale
import os
import platform
import struct
import sys
import typing as t

from ._version import get_versions

__all__ = ["show_versions"]


def get_system_info() -> t.Mapping[str, t.Any]:
    # Get git commit hash
    commit = get_versions()["full-revisionid"]  # type: t.Optional[str]
    version = get_versions()["version"]  # type: str

    size_integer = struct.calcsize("P")  # type: int

    return {
        "commit": commit,
        "Version": version,
        "python": sys.version,
        "python-bits": size_integer * 8,
        "OS": platform.system(),
        "OS-release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "byteorder": sys.byteorder,
        "LC_ALL": os.environ.get("LC_ALL"),
        "LANG": os.environ.get("LANG"),
        "LOCALE": ".".join(locale.getlocale()),
    }


def show_versions():
    """Display the versions of Pyxel and its dependencies."""

    system_info = get_system_info()  # type: t.Mapping[str, t.Any]

    dependencies_lst = (
        "pyxel",
        # required
        "astropy",
        "dask",
        "distributed",
        "h5py",
        "ipywidgets",
        "jupyter",
        "jupyterlab",
        "matplotlib",
        "numba",
        "numpy",
        "pandas",
        "poppy",
        "pygmo",
        # "pyyaml",
        "scipy",
        # Install / build
        "setuptools",
        "pip",
        "conda",
        # Test
        "black",
        "flake8",
        "isort",
        "mypy",
        "pytest",
        # Docs
        "sphinx",
    )  # type: t.Tuple[str, ...]

    dependencies = {}  # type: dict

    for module_name in dependencies_lst:
        try:
            # Try to get a module
            module = importlib.import_module(module_name)
        except Exception:
            dependencies[module_name] = None
        else:
            try:
                # Try to get a version
                version = getattr(module, "__version__")
                dependencies[module_name] = version
            except Exception:
                dependencies[module_name] = "installed"

    max_length = max(map(len, [*system_info, *dependencies]))
    print("")
    print("INSTALLED VERSIONS")
    print("------------------")
    for key, value in system_info.items():
        print(f"{key:<{max_length}}: {value}")
    print("")
    for key, value in dependencies.items():
        print(f"{key:<{max_length}}: {value}")
