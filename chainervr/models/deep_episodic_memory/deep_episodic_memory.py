#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.links as L
import chainer.functions as F

from .deep_episodic_memory_encoder import DeepEpisodicMemoryEncoder
from .deep_episodic_memory_decoder import DeepEpisodicMemoryDecoder


def resize_seq_images(x, size):
    """Resize sequential images"""

    assert x.ndim == 5  # BNCHW
    assert len(size) == 2  # H,W
    in_size = x.shape[3:]

    # BNCHW -> NBCHW
    x = x.transpose((1, 0, 2, 3, 4))
    nframes = x.shape[0]

    in_data = []
    for i in range(nframes):
        xi = x[i]
        if in_size != size:
            xi = F.resize_images(xi, size)
        xi = xi[:, None, :, :, :]
        in_data.append(xi)

    in_data = F.concat(in_data, axis=1)

    return in_data


class DeepEpisodicMemory(chainer.Chain):
    """Composite model for Deep Episodic Memory Network
       https://arxiv.org/pdf/1801.04134.pdf
    """

    def __init__(self, hidden_channels=None, num_episodes=None, dropout=None, noise_sigma=None):
        super(DeepEpisodicMemory, self).__init__()

        if noise_sigma is None:
            noise_sigma = 0.1

        if num_episodes is None:
            num_episodes = 5

        with self.init_scope():
            self.encoder = DeepEpisodicMemoryEncoder(
                out_channels=hidden_channels, dropout=dropout)
            self.decoder_reconst = DeepEpisodicMemoryDecoder(
                in_channels=hidden_channels, dropout=dropout, num_episodes=num_episodes)
            self.decoder_pred = DeepEpisodicMemoryDecoder(
                in_channels=hidden_channels, dropout=dropout, num_episodes=num_episodes)

        self.noise_sigma = noise_sigma
        self.num_episodes = num_episodes

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder_reconst.reset_state()
        self.decoder_pred.reset_state()

    def __call__(self, in_data):
        """in_data: (B, N, C, H, W)"""

        assert in_data.ndim == 5  # BNCHW

        xp = self.xp
        batch_size, nframes, nchannels = in_data.shape[:3]
        in_size = in_data.shape[3:]

        assert nframes == self.num_episodes, "%s != %s" % (self.num_episodes, nframes)

        self.reset_state()

        x = resize_seq_images(in_data, (128, 128))

        hidden = self.encoder(x)

        # add gaussian noise
        if chainer.config.train:
            noise_sigma = xp.log(self.noise_sigma ** 2, dtype=hidden.dtype)
            ln_var = xp.ones_like(hidden, dtype=hidden.dtype) * noise_sigma
            hidden = F.gaussian(hidden, ln_var)

        reconst = self.decoder_reconst(hidden)
        pred = self.decoder_pred(hidden)

        reconst = resize_seq_images(reconst, in_size)
        pred = resize_seq_images(pred, in_size)

        assert reconst.shape == in_data.shape
        assert pred.shape == in_data.shape

        return reconst, pred, hidden
