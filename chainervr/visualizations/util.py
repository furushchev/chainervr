#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.functions as F
import numpy as np


def extract_images(data):
    """Extract image data from array or chainer.Variable"""

    if isinstance(data, list):
        data = [extract_images(d) for d in data]

    if isinstance(data, chainer.Variable):
        data = F.copy(data, -1)
        data = data.array

    if data.ndim > 5:
        raise ValueError("invalid data: data.ndim > 5")
    elif data.ndim == 5:
        data = data[0]  # NCHW

    channels = data.shape[1]
    if channels == 1:  # mono
        out_shape = list(data.shape)
        out_shape[1] = 3
        data = np.broadcast_to(data, out_shape)

    data.flags.writeable = True

    # clip element
    data[data <= 0.0] = 0.0
    data[data >= 1.0] = 1.0
    return data
