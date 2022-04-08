from setuptools import setup, find_packages
import re

NAME = "mlcg"

# read the version number from the library
pattern = r"[0-9]\.[0-9]\.[0-9]"
VERSION = None
with open("./mlcg/__init__.py", "r") as fp:
    for line in fp.readlines():
        if "__version__" in line:
            VERSION = re.findall(pattern, line)[0]
if VERSION is None:
    raise ValueError("Version number not found.")


with open("requirements.txt") as f:
    install_requires = list(
        filter(lambda x: "#" not in x, (line.strip() for line in f))
    )

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(),
    zip_safe=True,
    python_requires=">=3.8",
    license="MIT",
    author="Fe"
    + "\u0301"
    + "lix Musil, Nick Charron, Yoayi Chen, Atharva Kelkar, Clark Templeton",
    install_requires=install_requires,
    scripts=[
        "scripts/mlcg-train.py",
        "scripts/mlcg-nvt_langevin.py",
        "scripts/mlcg-nvt_pt_langevin.py",
        "scripts/mlcg-combine_model.py",
        "scripts/mlcg-train_h5.py",
    ],
)
