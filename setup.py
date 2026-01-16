from setuptools import setup, find_packages

# Read README for long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="gapa",
    version="0.1.0",  # Major update: Unified Workflow API
    author="NetAlsGroup",
    author_email="netalsgroup@gmail.com",
    description="A PyTorch library for accelerating Genetic Algorithm in Perturbed SubStructure Optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NetAlsGroup/GAPA",
    license="MIT",
    
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
    ],
    
    python_requires=">=3.9",
    
    # Only include gapa package (not server/, web/, etc.)
    packages=find_packages(include=["gapa", "gapa.*"]),
    
    # Core dependencies (minimal for S/SM/M modes)
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "networkx>=2.8.0",
        "tqdm>=4.48.0",
    ],
    
    # Optional dependencies
    extras_require={
        # Full dependencies for all features
        "full": [
            "pandas>=2.0.0",
            "matplotlib>=3.5.0",
            "scipy>=1.10.0",
            "python-igraph>=0.11.0",
            "yacs>=0.1.8",
        ],
        # Deep learning attack algorithms
        "attack": [
            "dgl>=1.1.0",
            "deeprobust>=0.2.8",
        ],
        # Distributed mode (source deployment recommended)
        "distributed": [
            "requests>=2.28.0",
            "flask>=2.0.0",
        ],
        # Development dependencies
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
    },
    
    # Package metadata
    project_urls={
        "Homepage": "https://github.com/NetAlsGroup/GAPA",
        "Documentation": "https://github.com/NetAlsGroup/GAPA#readme",
        "Issues": "https://github.com/NetAlsGroup/GAPA/issues",
        "Changelog": "https://github.com/NetAlsGroup/GAPA/releases",
    },
    
    # Entry points (optional CLI)
    # entry_points={
    #     "console_scripts": [
    #         "gapa=gapa.cli:main",
    #     ],
    # },
    
    # Include package data
    include_package_data=True,
    zip_safe=False,
)