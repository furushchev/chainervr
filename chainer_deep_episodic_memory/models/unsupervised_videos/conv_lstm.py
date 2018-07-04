#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
from chainer import reporter
import chainer.functions as F
import chainer.links as L
from ... import links


class UnsupervisedLearningConvLSTM(chainer.Chain):
    def __init__(self, in_channels, out_channels,
                 in_episodes=None, out_episodes=None):
        super(UnsupervisedLearningConvLSTM, self).__init__()

        if n_layers is None:
            n_layers = 1

        self.encoder = []
        self.reconst = []
        self.pred = []

        for i in range(n_layers):
            pass

if __name__ == '__main__':
    pass
