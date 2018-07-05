#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.functions as F
from chainer.reporter import report

class EpisodicTrainChain(chainer.Chain):
    def __init__(self, model, ratio=None, scale=None, loss_func=None):
        super(EpisodicTrainChain, self).__init__()

        with self.init_scope():
            self.model = model

        if ratio is None:
            ratio = 0.5
        if scale is None:
            scale = 1.0
        if loss_func is None:
            loss_func = F.mean_squared_error
        self.ratio = ratio
        self.scale = scale
        self.loss_func = loss_func

    def __call__(self, in_data, t_data):
        xp = self.xp
        out_data = self.model(in_data)
        reconst, pred = out_data[0], out_data[1]

        reconst_loss, pred_loss = 0.0, 0.0

        if pred is not None:
            assert t_data.shape == pred.shape

        reconst_loss = self.loss_func(in_data, reconst)
        if pred is not None:
            pred_loss = self.loss_func(t_data, pred)

        loss = reconst_loss * self.ratio + pred_loss * (1.0 - self.ratio)
        loss *= self.scale

        report({
            "loss/reconst": reconst_loss,
            "loss/pred": pred_loss,
            "loss": loss
        }, self)

        return loss
