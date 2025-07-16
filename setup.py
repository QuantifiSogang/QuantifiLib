from setuptools import setup, find_packages

setup(
    name="quantifilib",
    version="0.1.0",
    author="Thomas Lee",
    description="A modular quantitative finance research library",
    packages=find_packages(),
    install_requires=[
        "pandas", "numpy", "yfinance", "pandas-datareader", "matplotlib"
    ],
    python_requires=">=3.8",
)