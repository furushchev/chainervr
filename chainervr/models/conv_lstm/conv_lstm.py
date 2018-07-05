#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
from chainer import reporter
import chainer.functions as F
import chainer.links as L
from chainer.utils import get_conv_outsize

from ... import links


class ConvLSTM(chainer.Chain):
    def __init__(self, n_channels, hidden_channels=None,
                 in_episodes=None, out_episodes=None,
                 patch_size=(128, 128),
                 predict=True):
        super(ConvLSTM, self).__init__()

        if hidden_channels is None:
            hidden_channels = [256, 128, 128]

        assert isinstance(n_channels, int)
        assert isinstance(hidden_channels, list)

        n_layers = len(hidden_channels)

        self.encoder = []
        self.reconst = []
        self.pred = []

        in_ch = n_channels
        for i in range(n_layers):
            out_ch = hidden_channels[i]
            print("lstm%d: %d -> %d" % (i, in_ch, out_ch))

            l = links.ConvolutionLSTM2D(in_ch, out_ch, 5)
            self.add_link("encoder%d" % i, l)
            self.encoder.append(l)

            l = links.ConvolutionLSTM2D(in_ch, out_ch, 5)
            self.add_link("reconst%d" % i, l)
            self.reconst.append(l)

            if predict:
                l = links.ConvolutionLSTM2D(in_ch, out_ch, 5)
                self.add_link("pred%d" % i, l)
                self.pred.append(l)

            in_ch = out_ch

        # fcLSTM
        l = L.Convolution2D(sum(hidden_channels), n_channels, 1)
        self.add_link("reconst_fc", l)

        if predict:
            l = L.Convolution2D(sum(hidden_channels), n_channels, 1)
            self.add_link("pred_fc", l)

        self.predict = predict
        self.n_layers = n_layers
        self.n_channels = n_channels
        self.patch_size = patch_size
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
            if self.patch_size is None:
                xi = x[i]
            else:
                xi = F.resize_images(x[i], self.patch_size)
            for e in self.encoder:
                hi = e(xi)
                xi = hi

        self.copy_state()

        # decode (reconstruct)
        reconst_imgs = []
        with chainer.cuda.get_device_from_id(self._device_id):
            if self.patch_size is None:
                xi = chainer.Variable(
                    xp.zeros((batch_size, nchannels,) + in_size, dtype=x.dtype))
            else:
                xi = chainer.Variable(
                    xp.zeros((batch_size, nchannels,) + self.patch_size, dtype=x.dtype))
        for i in range(self.in_episodes):
            rs = []
            for r in self.reconst:
                ri = r(xi)
                xi = ri
                rs.append(ri)
            ri = F.concat(rs)
            ri = self.reconst_fc(ri)
            ri = F.resize_images(ri, in_size)
            xi = ri
            reconst_imgs.append(ri[:, xp.newaxis])
        # B, (1, C, H, W) -> BNCHW
        reconst_imgs = F.concat(reconst_imgs, axis=1)

        # decode (prediction)
        pred_imgs = None
        if self.predict:
            pred_imgs = []
            with chainer.cuda.get_device_from_id(self._device_id):
                if self.patch_size is None:
                    xi = chainer.Variable(
                        xp.zeros((batch_size, nchannels,) + in_size, dtype=x.dtype))
                else:
                    xi = chainer.Variable(
                        xp.zeros((batch_size, nchannels,) + self.patch_size, dtype=x.dtype))
            for i in range(self.out_episodes):
                ps = []
                for p in self.pred:
                    pi = p(xi)
                    xi = pi
                    ps.append(pi)
                pi = F.concat(ps)
                pi = self.pred_fc(pi)
                pi = F.resize_images(pi, in_size)
                xi = pi
                pred_imgs.append(pi[:, xp.newaxis])
            pred_imgs = F.concat(pred_imgs, axis=1)

        return reconst_imgs, pred_imgs
