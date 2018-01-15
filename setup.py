# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages

# meta info
NAME = "dtwpy"
VERSION = "0.0.1"
AUTHOR = "Takehiro Suzuki"
AUTHOR_EMAIL = ""
URL = ""
DESCRIPTION = 'comprehensive dtw package for python'


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
    )


if __name__ == '__main__':
    main()
