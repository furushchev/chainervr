#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
import numpy as np
from .. import utils


url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"


def get_moving_mnist():
    return utils.cache_or_load_file(
        "moving_mnist", url, os.path.basename(url), np.load)


if __name__ == '__main__':
    data = get_moving_mnist()
    print(data.shape)
