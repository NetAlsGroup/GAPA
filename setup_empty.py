"""
Legacy minimal setup file.

Preferred modern paths:
    pip install .
    pip install -e .
    pip install ".[full]"
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
