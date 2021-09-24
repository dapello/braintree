#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "pytorch-lightning==1.2.10",
    "torch==1.8.1",
    "torchvision",
    "numpy",
    "pandas",
    "h5py",
    "cornet @ git+https://github.com/dicarlolab/CORnet",
    #"vonenet @ git+https://github.com/dicarlolab/vonenet"
]

setup(
    name='braintree',
    version='0.2.0',
    description="Modeling the Neural Mechanisms of Core Object Recognition",
    long_description=readme,
    author="Joel Dapello, Ko Kar",
    author_email='dapello@mit.edu, kohitij@mit.edu',
    url='https://github.com/dapello/braintree',
    packages=['braintree'],
    include_package_data=True,
    install_requires=requirements,
    license="GNU GPL v3",
    zip_safe=False,
    keywords='braintree Brain-Score',
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU GPL v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7'
    ],
)
