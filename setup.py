import setuptools
from pathlib import Path

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

setuptools.setup(
    name="ML model to cloud",
    version="0.0.0",
    author_email="ricardraigada@outlook.es",
    description="Deploying a ML model to cloud application platform\
        with FastAPI",
    author="Ricard Santiago Raigada García",
    url="https://github.com/ToroData/Deploying-a-ML-Model-to-Cloud-Application-Platform-with-FastAPI",
    packages=setuptools.find_namespace_packages(),
    install_requires=[required_packages],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8'
)
