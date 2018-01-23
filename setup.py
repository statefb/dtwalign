# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

# meta info
NAME = "dtwalign"
VERSION = "0.0.1"
AUTHOR = "Takehiro Suzuki"
AUTHOR_EMAIL = ""
URL = ""
DESCRIPTION = 'DTW package for python which enables partial alignment'
LICENSE = "MIT"


def main():

    setup(
        name=NAME,
        version=VERSION,
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        url=URL,
        description=DESCRIPTION,
        zip_safe=False,
        include_package_data=True,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "matplotlib",
            "seaborn >= 0.8.1",
            "networkx",
            "numba >= 0.34.0",
            "scipy"
        ],
        dependency_links = [

        ],
        tests_require=[],
        setup_requires=[],
        license=LICENSE,
        classifiers = [
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: 3.6",
            "License :: OSI Approved :: MIT License",
            "Topic :: Scientific/Engineering",
        ]
    )


if __name__ == '__main__':
    main()
