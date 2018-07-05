#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.links as L
import chainer.functions as F

from .deep_episodic_memory_encoder import DeepEpisodicMemoryEncoder
from .deep_episodic_memory_decoder import DeepEpisodicMemoryDecoder


class DeepEpisodicMemory(chainer.Chain):
    """Composite model for Deep Episodic Memory Network
       https://arxiv.org/pdf/1801.04134.pdf
    """

    def __init__(self, hidden_channels=None, num_episodes=None, dropout=None, noise_sigma=None):
        super(DeepEpisodicMemory, self).__init__()

        if noise_sigma is None:
            noise_sigma = 0.1

        with self.init_scope():
            self.encoder = DeepEpisodicMemoryEncoder(
                out_channels=hidden_channels, dropout=dropout)
            self.decoder_reconst = DeepEpisodicMemoryDecoder(
                in_channels=hidden_channels, dropout=dropout)
            self.decoder_pred = DeepEpisodicMemoryDecoder(
                in_channels=hidden_channels, dropout=dropout)

        self.noise_sigma = noise_sigma
        self.num_episodes = num_episodes

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder_reconst.reset_state()
        self.decoder_pred.reset_state()

    def __call__(self, x):
        """x: (B, N, C, H, W)"""

        xp = self.xp
        batch_size, nframes, nchannels = x.shape[:3]
        in_size = x.shape[3:]

        if self.num_episodes is not None:
            self.num_episodes = nframes
        else:
            assert nframes == self.num_episodes

        self.reset_state()

        # BNCHW -> NBCHW
        x = x.transpose((1, 0, 2, 3, 4))

        reconst_imgs = []
        pred_imgs = []
        hidden = None
        for i in range(self.num_episodes):
            xi = x[i]  # BCHW
            if in_size != (128, 128):
                xi = F.resize_images(xi, (128, 128))
            hi = self.encoder(xi)  # B, h_ch

            # add gaussian noise
            if chainer.config.train:
                noise_sigma = xp.log(self.noise_sigma ** 2, dtype=hi.dtype)
                ln_var = xp.ones_like(hi, dtype=hi.dtype) * noise_sigma
                hi = F.gaussian(hi, ln_var)

            ri = self.decoder_reconst(hi)  # B, CHW
            if in_size != (128, 128):
                ri = F.resize_images(ri, in_size)
            reconst_imgs.append(ri[:, xp.newaxis])

            pi = self.decoder_pred(hi)  # B, CHW
            if in_size != (128, 128):
                pi = F.resize_images(pi, in_size)
            pred_imgs.append(pi[:, xp.newaxis])

            if i == self.num_episodes - 1:
                hidden = hi

        reconst_imgs = F.concat(reconst_imgs, axis=1)
        pred_imgs = F.concat(pred_imgs, axis=1)

        return reconst_imgs, pred_imgs, hidden
