"""
Pyxel detector simulation framework.
"""

from setuptools import setup, find_packages
import pyxel


setup(
    name=pyxel.__appname__,
    version=pyxel.__version__,
    description=pyxel.__doc__,
    long_description=open("README.rst").read(),
    author=pyxel.__author__,
    author_email=pyxel.__author_email__,
    url="http://www.sci.esa.int/pyxel",
    license="MIT",
    keywords="ESA",
    install_requires=[
        "numpy",
        "astropy",
        "pandas",
        "scipy",
        "pygmo==2.10",
        "numba",
        "tqdm",
        "matplotlib",
        "tables",
        "Pillow",
        "poppy==0.8.0",
        "esapy_config",
    ],
    packages=find_packages(exclude=["tests*"]),
    include_package_data=True,
    package_data={"": ["*.glade", "*.ui", "*.acf"]},
    entry_points={"console_scripts": []},
)
