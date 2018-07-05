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

    def __init__(self, hidden_channels, out_channels,
                 encoder_cls=None, decoder_cls=None, episode_size=None):
        super(DeepEpisodicMemory, self).__init__()

        if encoder_cls is None:
            encoder_cls = DeepEpisodicMemoryEncoder
        if decoder_cls is None:
            decoder_cls = DeepEpisodicMemoryDecoder

        with self.init_scope():
            self.encoder = encoder_cls(fc_lstm_channels=hidden_channels)
            self.decoder_reconst = decoder_cls(
                fc_lstm_channels=hidden_channels, out_channels=out_channels)
            self.decoder_pred = decoder_cls(
                fc_lstm_channels=hidden_channels, out_channels=out_channels)

        if episode_size is None:
            episode_size = 5

        self.episode_size = episode_size

    def reset_state(self):
        self.encoder.reset_state()
        self.decoder_reconst.reset_state()
        self.decoder_pred.reset_state()

    def __call__(self, x):
        """x: (B, C, H, W)"""
        hidden = self.encoder(x)
        reconst = self.decoder_reconst(hidden)
        pred = self.decoder_pred(hidden)
        with chainer.cuda.get_device_from_id(self._device_id):
            pred_ret = chainer.Variable(pred.array.copy())
            reconst_ret = chainer.Variable(reconst.array.copy())
            hidden_ret = chainer.Variable(hidden.array.copy())
        return pred_ret, reconst_ret, hidden_ret
