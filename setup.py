import setuptools
import pathlib


def parse_requirements(filename="requirements.txt"):
    with pathlib.Path(filename).open() as requirements_file:
        return [
            line.strip()
            for line in requirements_file
            if line.strip() and not line.startswith("#")
        ]


setuptools.setup(
    name="RecurrentHDR",
    version="0.0.1",
    author="Sebastian Dille",
    author_email="sdille@sfu.ca",
    description='A package containing to the code for the paper "Recurrent Dynamic Range Extension", Proc. SIGGRAPH Asia, 2026',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    license="",
    install_requires=parse_requirements(),
    python_requires=">=3.10",
)
