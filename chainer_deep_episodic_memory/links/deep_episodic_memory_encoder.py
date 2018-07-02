#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.links as L
import chainer.functions as F

from .convolution_lstm_2d import ConvolutionLSTM2D
from .layer_normalization import LayerNormalization


class DeepEpisodicMemoryEncoder(chainer.Chain):
    """Deep Episodic Memory Encoder"""

    def __init__(self, fc_channels=1000, fc_lstm_channels=1000,
                 dropout_ratio=0.1):
        super(DeepEpisodicMemoryEncoder, self).__init__(
            conv1=L.Convolution2D(3, 32, 5, stride=2, pad=2),
            conv_norm1=LayerNormalization(),
            lstm1=ConvolutionLSTM2D(32, 32, 5),
            lstm_norm1=LayerNormalization(),
            #
            conv2=L.Convolution2D(32, 32, 5, stride=2, pad=2),
            conv_norm2=LayerNormalization(),
            lstm2=ConvolutionLSTM2D(32, 32, 5),
            lstm_norm2=LayerNormalization(),
            #
            conv3=L.Convolution2D(32, 32, 5, stride=2, pad=2),
            conv_norm3=LayerNormalization(),
            lstm3=ConvolutionLSTM2D(32, 32, 3),
            lstm_norm3=LayerNormalization(),
            #
            conv4=L.Convolution2D(32, 32, 3, stride=2, pad=1),
            conv_norm4=LayerNormalization(),
            lstm4=ConvolutionLSTM2D(32, 64, 3),
            lstm_norm4=LayerNormalization(),
            #
            conv5=L.Convolution2D(64, 64, 3, stride=2, pad=1),
            conv_norm5=LayerNormalization(),
            lstm5=ConvolutionLSTM2D(64, 64, 3),
            lstm_norm5=LayerNormalization(),
            #
            fc_conv=L.Convolution2D(64, fc_channels, 4, stride=1, pad=0),
            fc_lstm=ConvolutionLSTM2D(fc_channels, fc_lstm_channels, 1),
        )
        self.dropout_ratio = dropout_ratio
        self.reset_state()

    def reset_state(self):
        for link in self.links():
            if link != self and hasattr(link, "reset_state"):
                link.reset_state()

    def __call__(self, x):
        h = self.conv1(x)
        h = F.relu(self.conv_norm1(h))
        #
        h = self.lstm1(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm1(h)
        #
        h = self.conv2(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.conv_norm2(h))
        #
        h = self.lstm2(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm2(h)
        #
        h = self.conv3(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.conv_norm3(h))
        #
        h = self.lstm3(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm3(h)
        #
        h = self.conv4(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.conv_norm4(h))
        #
        h = self.lstm4(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm4(h)
        #
        h = self.conv5(F.dropout(h, ratio=self.dropout_ratio))
        h = F.relu(self.conv_norm5(h))
        #
        h = self.lstm5(F.dropout(h, ratio=self.dropout_ratio))
        h = self.lstm_norm5(h)
        #
        h = self.fc_conv(F.dropout(h, ratio=self.dropout_ratio))
        h = self.fc_lstm(F.dropout(h, ratio=self.dropout_ratio))
        #
        return F.concat((self.fc_lstm.c, self.fc_lstm.h))  # 2 * fc_lstm_channels
