#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.links as L
import chainer.functions as F

from ... import links


class DeepEpisodicMemoryDecoder(chainer.Chain):
    def __init__(self, out_channels, fc_lstm_channels=1000,
                 dropout_ratio=0.1):
        super(DeepEpisodicMemoryDecoder, self).__init__(
            fc_lstm=links.ConvolutionLSTM2D(
                fc_lstm_channels, fc_lstm_channels, 1, pad=0),
            fc_deconv=L.Deconvolution2D(
                fc_lstm_channels, 64, 4, stride=1, pad=0),
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
                32, out_channels, 5, stride=2, pad=2, outsize=(128, 128)),
        )

        self.fc_lstm_channels = fc_lstm_channels
        self.dropout_ratio = dropout_ratio
        self.reset_state()

    def reset_state(self):
        for link in self.links():
            if link != self and hasattr(link, "reset_state"):
                link.reset_state()

    def __call__(self, x):
        assert isinstance(x, chainer.Variable)
        assert x.shape[1] == self.fc_lstm_channels * 2
        c, h = F.split_axis(x, 2, axis=1)
        self.fc_lstm.c, self.fc_lstm.h = c, h

        h = self.fc_lstm(h)
        h = self.fc_deconv(F.dropout(h, ratio=self.dropout_ratio))
        #
        h = self.lstm1(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm1(h)
        #
        h = self.deconv1(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.deconv_norm1(h))
        #
        h = self.lstm2(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm2(h)
        #
        h = self.deconv2(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.deconv_norm2(h))
        #
        h = self.lstm3(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm3(h)
        #
        h = self.deconv3(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.deconv_norm3(h))
        #
        h = self.lstm4(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm4(h)
        #
        h = self.deconv4(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.deconv_norm4(h))
        #
        h = self.lstm5(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm5(h)
        #
        o = self.deconv5(F.dropout(h, ratio=self.dropout_ratio))
        #
        return o
