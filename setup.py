from setuptools import setup, find_packages

setup(
    name="my_mlp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.4.0",
        "tqdm>=4.65.0",
        "PyQt6>=6.4.0",
        "pandas>=1.3.0"
    ],
    description="A neural network implementation from scratch with backpropagation",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.7",
) 