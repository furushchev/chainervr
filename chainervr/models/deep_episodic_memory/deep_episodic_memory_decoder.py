#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.links as L
import chainer.functions as F

from ... import links


class DeepEpisodicMemoryDecoder(chainer.Chain):
    """Deep Episodic Memory Decoder"""

    def __init__(self, num_episodes, in_channels=None, dropout=None):
        if in_channels is None:
            in_channels = 1000
        if dropout is None:
            dropout = 0.1

        super(DeepEpisodicMemoryDecoder, self).__init__(
            fc_lstm=links.ConvolutionLSTM2D(
                in_channels, in_channels, 1, pad=0),
            fc_deconv=L.Deconvolution2D(
                in_channels, 64, 4, stride=1, pad=0),
            #
            lstm1=links.ConvolutionLSTM2D(64, 64, 3),
            lstm_norm1=links.LayerNormalization(),
            deconv1=L.Deconvolution2D(
                64, 64, 3, stride=2, pad=1, outsize=(8, 8)),
            deconv_norm1=links.LayerNormalization(),
            #
            lstm2=links.ConvolutionLSTM2D(64, 64, 3),
            lstm_norm2=links.LayerNormalization(),
            deconv2=L.Deconvolution2D(
                64, 64, 3, stride=2, pad=1, outsize=(16, 16)),
            deconv_norm2=links.LayerNormalization(),
            #
            lstm3=links.ConvolutionLSTM2D(64, 32, 3),
            lstm_norm3=links.LayerNormalization(),
            deconv3=L.Deconvolution2D(
                32, 32, 5, stride=2, pad=2, outsize=(32, 32)),
            deconv_norm3=links.LayerNormalization(),
            #
            lstm4=links.ConvolutionLSTM2D(32, 32, 5),
            lstm_norm4=links.LayerNormalization(),
            deconv4=L.Deconvolution2D(
                32, 32, 5, stride=2, pad=2, outsize=(64, 64)),
            deconv_norm4=links.LayerNormalization(),
            #
            lstm5=links.ConvolutionLSTM2D(32, 32, 5),
            lstm_norm5=links.LayerNormalization(),
            deconv5=L.Deconvolution2D(
                32, 3, 5, stride=2, pad=2, outsize=(128, 128)),
        )

        self.in_channels = in_channels
        self.dropout = dropout
        self.num_episodes = num_episodes

    def reset_state(self):
        for link in self.links():
            if link != self and hasattr(link, "reset_state"):
                link.reset_state()

    def __call__(self, x):
        xp = self.xp
        assert x.ndim == 4, "%s != 4" % (x.ndim)  # B,2C,1,1
        assert x.shape[1] == self.in_channels * 2
        assert x.shape[2] == x.shape[3] == 1

        c0, h0 = F.split_axis(x, 2, axis=1)
        self.fc_lstm.set_state(c0, h0)
        l0 = h0

        outputs = []
        for i in range(self.num_episodes):
            l0 = self.fc_lstm(l0)
            d0 = self.fc_deconv(F.dropout(l0, ratio=self.dropout))
            #
            l1 = self.lstm1(F.dropout(d0, ratio=self.dropout))
            l1 = self.lstm_norm1(l1)
            #
            d1 = self.deconv1(F.dropout(l1, ratio=self.dropout))
            d1 = F.relu(self.deconv_norm1(d1))
            #
            l2 = self.lstm2(F.dropout(d1, ratio=self.dropout))
            l2 = self.lstm_norm2(l2)
            #
            d2 = self.deconv2(F.dropout(l2, ratio=self.dropout))
            d2 = F.relu(self.deconv_norm2(d2))
            #
            l3 = self.lstm3(F.dropout(d2, ratio=self.dropout))
            l3 = self.lstm_norm3(l3)
            #
            d3 = self.deconv3(F.dropout(l3, ratio=self.dropout))
            d3 = F.relu(self.deconv_norm3(d3))
            #
            l4 = self.lstm4(F.dropout(d3, ratio=self.dropout))
            l4 = self.lstm_norm4(l4)
            #
            d4 = self.deconv4(F.dropout(l4, ratio=self.dropout))
            d4 = F.relu(self.deconv_norm4(d4))
            #
            l5 = self.lstm5(F.dropout(d4, ratio=self.dropout))
            l5 = self.lstm_norm5(l5)
            #
            o = self.deconv5(F.dropout(l5, ratio=self.dropout))
            #
            o = o[:, None, :, :, :]
            outputs.append(o)  # <- B1CHW

        outputs = F.concat(outputs, axis=1)  # BNCHW

        return outputs
