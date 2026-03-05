from setuptools import setup, find_packages

setup(
    name="XdetectionCore",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # Crucial to include the .mplstyle file
    package_data={'': ['resources/*.mplstyle']},
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "joblib",
        "tqdm"
    ],
    description="Foundational utilities for the Akrami Lab Xdetection pipeline.",
)