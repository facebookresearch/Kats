#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

from setuptools import find_packages, setup


# read long description from README
with open("README.md", "r") as f:
    long_description = f.read()

# read dependency requirements
with open("requirements.txt", "r") as f:
    install_requires = f.read().splitlines()

# optional dependencies skipped when MINIMAL_KATS=1
with open("test_requirements.txt", "r") as f:
    extra_requires = f.read().splitlines()

if not os.environ.get("MINIMAL_KATS", False):
    install_requires += extra_requires


setup(
    name="kats",
    packages=find_packages(),
    version="0.2.0",
    license="MIT",
    description="kats: kit to analyze time series",
    author="facebook",
    author_email="iamxiaodong@fb.com",
    url="https://github.com/facebookresearch/Kats",
    download_url="https://github.com/facebookresearch/Kats/archive/refs/tags/v0.2.tar.gz",
    keywords=[
        "time series",
        "forecasting",
        "anomaly detection",
        "tsfeatures",
        "temporal embedding",
    ],
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
