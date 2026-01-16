"""
GAPA Minimal Setup
==================
Minimal setup file for development installation.
For full setup and PyPI publishing, use setup.py instead.

Usage:
    pip install -e .  (using setup.py)
    pip install .     (using setup.py)
"""

from setuptools import setup, find_packages

setup(
    name="gapa",
    version="0.1.0",
    description="A PyTorch library for accelerating Genetic Algorithm in Perturbed SubStructure Optimization.",
    packages=find_packages(include=["gapa", "gapa.*"]),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "networkx>=2.8.0",
        "tqdm>=4.48.0",
    ],
    python_requires=">=3.9",
)