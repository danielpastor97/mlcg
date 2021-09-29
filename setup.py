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
    "pytest",
]


setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    zip_safe=True,
    python_requires=">=3.8",
    license="MIT",
    author="Fe" + "\u0301" + "lix Musil, Nick Charron",
    install_requires=install_requires,
)
