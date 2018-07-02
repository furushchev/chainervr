#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

from setuptools import find_packages
from setuptools import setup

import chainer_deep_episodic_memory as pkg

author_raw = pkg.__author__.replace("<", "").replace(">", "").split()
author = " ".join(author_raw[:-1]).strip()
author_email = author_raw[-1].strip()

setup(
    name='chainer-deep-episodic-memory',
    version=pkg.__version__,
    packages=find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').readlines(),
    author=author,
    author_email=author_email,
    description="Chainer implementation of Deep Episodic Memory",
    url=pkg.__url__,
    license="MIT",
)
