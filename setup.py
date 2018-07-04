#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from setuptools import find_packages
from setuptools import setup


setup(
    name='chainer-deep-episodic-memory',
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').readlines(),
    author="Yuki Furuta",
    author_email="furushchev@jsk.imi.i.u-tokyo.ac.jp",
    description="Chainer implementation of Deep Episodic Memory",
    url="https://github.com/furushchev/chainer-deep-episodic-memory",
    license="MIT",
)
