#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.functions as F

class EpisodicTrainChain(chainer.Chain):
    def __init__(self, model, ratio=None, scale=None):
        super(EpisodicTrainChain, self).__init__()

        with self.init_scope():
            self.model = model

        if ratio is None:
            ratio = 0.5
        if scale is None:
            scale = 1.0
        self.ratio = ratio
        self.scale = scale

    def __call__(self, in_data, t_data):
        xp = self.xp
        reconst, pred = self.model(in_data)

        reconst_loss, pred_loss = 0.0, 0.0

        if pred is not None:
            assert t_data.shape == pred.shape

        if xp.issubdrype(in_data.dtype, xp.integer):
            # for integer
            reconst_loss = F.softmax_cross_entropy(in_data, reconst)
            if pred is not None:
                pred_loss = F.softmax_cross_entropy(t_data, pred)
        elif xp.issubdrype(in_data.dtype, xp.floating):
            # for float
            reconst_loss = F.mean_squared_error(in_data, reconst)
            if pred is not None:
                pred_loss = F.mean_squared_error(t_data, pred)
        else:
            raise TypeError("data must be int or float")

        loss = reconst_loss * self.ratio + pred_loss * (1.0 - self.ratio)
        loss *= self.scale

        reporter.report({
            "loss/reconst": reconst_loss,
            "loss/pred": pred_loss,
            "loss": loss
        }, self)

        return loss
