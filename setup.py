from setuptools import setup, find_packages

NAME = "mlcg"
VERSION = "0.1"

install_requires = [
    "mdtraj",
    "jsonargparse[signatures]",
    "tqdm",
    "ase",
    "e3nn",
    "numpy",
    "scipy",
    "scikit-learn",
]


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    zip_safe=True,
    python_requires=">=3.8",
    install_requires=install_requires,
)
