#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import numpy as np
from . import ucf101_utils as utils


class UCF101Dataset(chainer.dataset.DatasetMixin):
    def __init__(self, split="train",
                 dataset_num=None, num_episodes=None,
                 with_class=False):
        if num_episodes is None:
            num_episodes = 5

        dataset = utils.get_ucf101(split, n=dataset_num)

        self.root_dir = dataset["root_dir"]
        self.annotations = dataset["annotations"]
        self.num_episodes = num_episodes
        self.with_class = with_class

    def __len__(self):
        return self.annotations["filename"].shape[0]

    def get_example(self, i):
        videoname = self.annotations["filename"][i]
        cls = self.annotations["class"][i]
        data = utils.load_episode(videoname,
                                  num_episodes=self.num_episodes)
        if self.with_class:
            return data, cls
        else:
            return data


if __name__ == '__main__':
    import os
    if not os.getenv("DISPLAY", None):
        import matplotlib
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from chainervr.visualizations import vis_episode

    dataset = UCF101Dataset()

    fig = plt.figure()

    n = 10
    for i in range(n):
        idx = (len(dataset) // n) * i
        episode, cls = dataset[idx]
        cls_str = utils.ucf101_class_names[cls]
        ax = fig.add_subplot(n, 1, i+1)
        vis_episode(episode, ax=ax, title=cls_str, fontsize=4)

    plt.savefig("image.jpg", bbox_inches="tight", dpi=300)
    plt.close()
