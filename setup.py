#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from setuptools import setup, find_packages

# read long description from README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# read dependency requirements
with open('requirements.txt', 'r') as f:
    install_requires = f.read().splitlines()

setup(
  name = 'kats',
  packages = find_packages(),
  version = '0.1.0',
  license='MIT',
  description = 'kats: kit to analyze time series',
  author = 'facebook',
  author_email = 'iamxiaodong@fb.com',
  url = 'https://github.com/facebookresearch/Kats',
  download_url = 'https://github.com/facebookresearch/Kats/archive/refs/tags/v0.1.tar.gz',
  keywords = ['time series', 'forecasting', 'anomaly detection', 'tsfeatures', 'temporal embedding'],
  install_requires=install_requires,
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
  long_description=long_description,
  long_description_content_type='text/markdown',
)
