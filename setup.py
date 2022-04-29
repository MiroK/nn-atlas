#!/usr/bin/env python

from setuptools import find_packages, setup

setup(name = 'nn-atlas',
      version = '0.1',
      description = 'Learning extension operators',
      author = 'Miroslav Kuchta',
      author_email = 'miroslav.kuchta@gmail.com',
      url = 'https://github.com/mirok/nn-atlas.git',
      packages=find_packages(),
      include_package_data=True
)
