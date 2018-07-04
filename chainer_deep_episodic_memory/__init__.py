#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import pkg_resources


__dist__ = pkg_resources.get_distribution(__name__)

__version__ = __dist__.version


from . import datasets
from . import functions
from . import links
from . import models
from . import training
