# -*- coding: utf-8 -*-
from setuptools import setup
from setuptools import find_packages
# from msta_net import __author__,__version__

def main():
    description = 'fast dtw for python'

    setup(
        name='pyfastdtw',
        # version=__version__,
        # author=__author__,
        author_email='',
        url='',
        description=description,
        long_description=description,
        zip_safe=False,
        include_package_data=True,
        packages=find_packages(),
        install_requires=[
            "numpy",
            "matplotlib",
            "seaborn",
            "networkx"
        ],
        dependency_links = [

        ],
        tests_require=[],
        setup_requires=[],
    )


if __name__ == '__main__':
    main()
