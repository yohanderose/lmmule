import os
import re
from setuptools import setup, find_packages


def read_requirements() -> list:
    with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
        return f.read().splitlines()


setup(
    name="lmmule",
    version="0.1.1",
    author="Yohan de Rose",
    author_email="yohan@aapstaart.com",
    description=re.search(r"Lightweight.*?\.", open("README.md").read()).group(),
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yohanderose/lmmule",
    packages=find_packages(include=["lmmule", "lmmule.*"]),
    install_requires=read_requirements(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
