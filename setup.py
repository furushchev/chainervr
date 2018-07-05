#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from setuptools import find_packages
from setuptools import setup


setup(
    name='chainer-video-representations',
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').readlines(),
    author="Yuki Furuta",
    author_email="furushchev@jsk.imi.i.u-tokyo.ac.jp",
    description="Chainer implementation of Networks for Learning Visual Representation",
    url="https://github.com/furushchev/chainervr",
    license="MIT",
)
