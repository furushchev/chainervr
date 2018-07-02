#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.functions as F


class LayerNormalization(chainer.Link):
    """https://arxiv.org/pdf/1607.06450.pdf
       http://cs231n.stanford.edu/slides/2018/cs231n_2018_lecture07.pdf
    """

    def __init__(self, size=None, eps=1e-6, initial_gamma=None,
                 initial_beta=None):
        super(LayerNormalization, self).__init__()
        if initial_gamma is None:
            initial_gamma = 1
        if initial_beta is None:
            initial_beta = 0

        with self.init_scope():
            self.gamma = chainer.variable.Parameter(initial_gamma)
            self.beta = chainer.variable.Parameter(initial_beta)
            self.eps = eps

        if size is not None:
            self._initialize_params(size)

    def _initialize_params(self, size):
        self.gamma.initialize(size)
        self.beta.initialize(size)

    def __call__(self, x):
        if self.gamma.data is None:
            in_size = x[0].shape
            self._initialize_params(in_size)
        mean = F.broadcast_to(F.mean(x, keepdims=True), x.shape)
        var = F.broadcast_to(F.mean((x - mean) ** 2, keepdims=True), x.shape)
        mean = mean[0]
        var = var[0]
        return F.fixed_batch_normalization(
            x, self.gamma, self.beta, mean, var, self.eps)
