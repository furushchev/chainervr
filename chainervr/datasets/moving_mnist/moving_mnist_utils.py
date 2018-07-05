#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
import numpy as np
import shutil
from chainer.dataset import download


root = "pfnet/chainervr/moving_mnist"
url = "http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"


def get_moving_mnist():
    data_dir = download.get_dataset_directory(root)
    path = os.path.join(data_dir, os.path.basename(url))
    def cached_download(dst):
        src = download.cached_download(url)
        shutil.copy(src, dst)
        return np.load(dst)
    return download.cache_or_load_file(
        path,
        lambda path: cached_download(path),
        np.load)
