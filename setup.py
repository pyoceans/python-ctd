from pathlib import Path

from setuptools import find_packages, setup

import versioneer

rootpath = Path(__file__).parent.absolute()


def read(*parts):
    return open(rootpath.joinpath(*parts), "r").read()


with open("requirements.txt") as f:
    requires = f.readlines()
install_requires = [req.strip() for req in requires]

setup(
    name="ctd",
    python_requires=">=3.6",
    version=versioneer.get_version(),
    description="Tools to load hydrographic data as DataFrames",
    license="BSD-3-Clause",
    long_description=f'{read("README.md")}',
    long_description_content_type="text/markdown",
    author="Filipe Fernandes",
    author_email="ocefpaf@gmail.com",
    url="https://github.com/pyoceans/python-ctd",
    download_url="http://pypi.python.org/pypi/ctd",
    keywords=["oceanography", "CTD", "pandas"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: BSD License",
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    platforms="any",
    packages=find_packages(),
    extras_require={"testing": ["pytest"]},
    install_requires=install_requires,
    cmdclass=versioneer.get_cmdclass(),
)
