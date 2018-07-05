#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
from chainer import reporter
import chainer.functions as F
import chainer.links as L


class RPLSTM(chainer.Chain):
    def __init__(self, n_channels, patch_size, n_layers=None, n_hidden=None, predict=True,
                 in_episodes=None, out_episodes=None):
        super(RPLSTM, self).__init__()

        if n_layers is None:
            n_layers = 1
        if n_hidden is None:
            n_hidden = 2048

        n_units = n_channels * patch_size[0] * patch_size[1]

        in_units = n_units
        out_units = n_hidden
        self.encoder = []
        self.reconst = []
        self.pred = []
        for i in range(n_layers):
            if i == n_layers - 1:
                out_units = n_units

            print("lstm%d: %d -> %d" % (i, in_units, out_units))

            l = L.LSTM(in_units, out_units)
            self.add_link("encoder%d" % i, l)
            self.encoder.append(l)

            l = L.LSTM(in_units, out_units)
            self.add_link("reconst%d" % i, l)
            self.reconst.append(l)

            if predict:
                l = L.LSTM(in_units, out_units)
                self.add_link("pred%d" % i, l)
                self.pred.append(l)

            in_units = out_units

        self.predict = predict
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.n_units = n_units
        self.n_layers = n_layers
        self.in_episodes = in_episodes
        self.out_episodes = out_episodes

    def reset_state(self):
        for i in range(self.n_layers):
            self.encoder[i].reset_state()
            self.reconst[i].reset_state()
            if self.predict:
                self.pred[i].reset_state()

    def copy_state(self):
        for i in range(self.n_layers):
            e = self.encoder[i]
            self.reconst[i].set_state(e.c, e.h)
            if self.predict:
                self.pred[i].set_state(e.c, e.h)

    def __call__(self, x):
        xp = self.xp
        batch_size, nframes, nchannels = x.shape[:3]
        in_size = x.shape[3:]

        if self.in_episodes is None:
            self.in_episodes = self.out_episodes = nframes
        else:
            assert self.in_episodes == nframes

        self.reset_state()

        # BNCHW -> NBCHW
        x = x.transpose((1, 0, 2, 3, 4))

        # encode
        for i in range(self.in_episodes):
            xi = F.resize_images(x[i], self.patch_size)
            xi = xi.reshape((batch_size, -1))
            for e in self.encoder:
                hi = e(xi)
                xi = hi

        self.copy_state()

        # decode (reconstruct)
        reconst_imgs = []
        with chainer.cuda.get_device_from_id(self._device_id):
            xi = chainer.Variable(xp.zeros_like(xi, dtype=xi.dtype))
        for i in range(self.in_episodes):
            for r in self.reconst:
                ri = r(xi)
                xi = ri
            ri = ri.reshape((batch_size, self.n_channels) + self.patch_size)  # BCHW
            ri = F.resize_images(ri, in_size)
            reconst_imgs.append(ri[:, xp.newaxis])  # B, 1, C, H, W

        reconst_imgs = F.concat(reconst_imgs, axis=1) # BFCHW

        # decode (prediction)
        pred_imgs = None
        if self.predict:
            pred_imgs = []
            with chainer.cuda.get_device_from_id(self._device_id):
                xi = chainer.Variable(xp.zeros_like(xi, dtype=xi.dtype))
            for i in range(self.out_episodes):
                for p in self.pred:
                    pi = p(xi)
                    xi = pi
                pi = pi.reshape((batch_size, self.n_channels) + self.patch_size)
                pi = F.resize_images(pi, in_size)
                pred_imgs.append(pi[:, xp.newaxis])

            pred_imgs = F.concat(pred_imgs, axis=1)  # BFCHW

        return reconst_imgs, pred_imgs
