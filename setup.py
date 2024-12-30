from setuptools import setup, find_packages

setup(
    name="gapa",
    version="0.0.1",
    author="netalsgroup",
    author_email="xx@xx.xx",
    description="A PyTorch library for accelerating Genetic Algorithm in Perturbed SubStructure Optimization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/NetAlsGroup/GAPA",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.3",
        "pandas>=2.2.1",
        "matplotlib>=3.8.4",
        "scipy>=1.10.1",
        "networkx>=3.2.1",
        "igraph>=0.11.5",
        "tqdm>=4.48.2",
        "yacs>=0.1.8",
    ],
    extras_require={
        "cpu": [
            "torch>=2.3.0; sys_platform == 'darwin' or extra == 'cpu'",
            "dgl>=2.2.1; sys_platform == 'darwin' or extra == 'cpu'",
            "deeprobust>=0.2.10; sys_platform == 'darwin' or extra == 'cpu'",
        ],
        "cuda118": [
            "torch>=2.3.0; sys_platform == 'darwin' and extra == 'cuda'",
            "dgl-cu111>=0.6.1; sys_platform == 'darwin' and extra == 'cuda'",
            "deeprobust>=0.2.10; sys_platform == 'darwin' and extra == 'cuda'",
        ],
    },
    project_urls={
        "Homepage": "https://github.com/NetAlsGroup/GAPA",
        "Issues": "https://github.com/NetAlsGroup/GAPA/issues",
    },
)