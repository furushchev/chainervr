#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.functions as F
from ... import functions


class DeepEpisodicMemoryTrainChain(chainer.Chain):
    """Chain for Training Deep Episodic Memory Network
       https://arxiv.org/pdf/1801.04134.pdf
    """

    def __init__(self, model, loss_func="mse", mse_ratio=0.6, resize_images=None):
        super(DeepEpisodicMemoryTrainChain, self).__init__()
        with self.init_scope():
            self.model = model

        loss_func = loss_func.lower()
        loss_funcs = ["mse", "gdl", "mse_gdl"]
        if loss_func not in loss_funcs:
            raise ValueError(
                "loss_func '%s' must be either: %s" % (loss_func, loss_funcs))
        if mse_ratio < 0.0 or mse_ratio > 1.0:
            raise ValueError(
                "mse_ratio '%s' must be from 0.0 to 1.0" % mse_ratio)

        if resize_images is None:
            resize_images = (128, 128)

        self.loss_func = loss_func
        self.mse_ratio = mse_ratio
        self.episode_size = self.model.episode_size
        self.resize_images = resize_images

    def reset_state(self):
        self.model.reset_state()

    def __call__(self, x, t_reconst, t_pred):
        """x, t_reconst, t_pred: (B, C, H, W)"""

        if self.resize_images:
            x = F.resize_images(x, self.resize_images)
            t_reconst = F.resize_images(t_reconst, self.resize_images)
            t_pred = F.resize_images(t_pred, self.resize_images)
        pred, reconst, hidden = self.model(x)

        mse_loss = 0.
        if "mse" in self.loss_func:
            mse_loss += F.sum(F.mean_squared_error(pred, t_pred))
            mse_loss += F.sum(F.mean_squared_error(reconst, t_reconst))
            chainer.report({"loss/mse": mse_loss}, self)

        gdl_loss = 0.
        if "gdl" in self.loss_func:
            gdl_loss += functions.gradient_difference(
                pred, t_pred, alpha=2.0)
            gdl_loss += functions.gradient_difference(
                reconst, t_reconst, alpha=2.0)
            chainer.report({"loss/gdl": gdl_loss}, self)

        loss = mse_loss * self.mse_ratio + gdl_loss * (1.0 - self.mse_ratio)
        chainer.report({"loss": loss}, self)

        return loss
