#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import numpy as np
import chainer
from .moving_mnist_utils import get_moving_mnist


class MovingMnistDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_path="auto", channels_num=None, dtype=None,):
        if data_path == "auto":
            self.dataset = get_moving_mnist()
        else:
            self.dataset = np.load(data_path)

        self.dataset = self.dataset.transpose((1, 0, 2, 3))

        if dtype is None:
            dtype = np.float32
        self.dataset = self.dataset.astype(dtype)
        if channels_num is None:
            channels_num = 3
        self.channels_num = channels_num

    def __len__(self):
        return self.dataset.shape[0]

    def get_example(self, i):
        data = self.dataset[i]
        data = np.broadcast_to(data, (self.channels_num,) + data.shape)
        return data.transpose((1, 0, 2, 3))


if __name__ == '__main__':
    import os
    if not os.getenv("DISPLAY", None):
        import matplotlib
        matplotlib.use('Agg')
    from chainercv.visualizations import vis_image
    import matplotlib.pyplot as plt

    dataset = MovingMnistDataset()
    for i, frames in enumerate(dataset):
        for j, frame in enumerate(frames):
            print(frame.shape)
            vis_image(frame)
            fname = "image_%05d_%02d.jpg" % (i, j)
            plt.savefig(fname)
            print("Saved to %s" % fname)
            plt.show()
