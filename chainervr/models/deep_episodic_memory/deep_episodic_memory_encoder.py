#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
import chainer.links as L
import chainer.functions as F

from ... import links


class DeepEpisodicMemoryEncoder(chainer.Chain):
    """Deep Episodic Memory Encoder"""

    def __init__(self, out_channels=None, dropout=None):
        if out_channels is None:
            out_channels = 1000
        if dropout is None:
            dropout = 0.1

        super(DeepEpisodicMemoryEncoder, self).__init__(
            conv1=L.Convolution2D(3, 32, 5, stride=2, pad=2),
            conv_norm1=links.LayerNormalization(),
            lstm1=links.ConvolutionLSTM2D(32, 32, 5),
            lstm_norm1=links.LayerNormalization(),
            #
            conv2=L.Convolution2D(32, 32, 5, stride=2, pad=2),
            conv_norm2=links.LayerNormalization(),
            lstm2=links.ConvolutionLSTM2D(32, 32, 5),
            lstm_norm2=links.LayerNormalization(),
            #
            conv3=L.Convolution2D(32, 32, 5, stride=2, pad=2),
            conv_norm3=links.LayerNormalization(),
            lstm3=links.ConvolutionLSTM2D(32, 32, 3),
            lstm_norm3=links.LayerNormalization(),
            #
            conv4=L.Convolution2D(32, 32, 3, stride=2, pad=1),
            conv_norm4=links.LayerNormalization(),
            lstm4=links.ConvolutionLSTM2D(32, 64, 3),
            lstm_norm4=links.LayerNormalization(),
            #
            conv5=L.Convolution2D(64, 64, 3, stride=2, pad=1),
            conv_norm5=links.LayerNormalization(),
            lstm5=links.ConvolutionLSTM2D(64, 64, 3),
            lstm_norm5=links.LayerNormalization(),
            #
            fc_conv=L.Convolution2D(64, out_channels, 4, stride=1, pad=0),
            fc_lstm=links.ConvolutionLSTM2D(out_channels, out_channels, 1),
        )

        self.out_channels = out_channels
        self.dropout = dropout

    def reset_state(self):
        for link in self.links():
            if link != self and hasattr(link, "reset_state"):
                link.reset_state()

    def __call__(self, x):
        assert x.ndim == 5  # BNCHW

        # BNCHW -> NBCHW
        x = x.transpose((1, 0, 2, 3, 4))

        nframes = x.shape[0]

        for i in range(nframes):
            c1 = self.conv1(x[i])
            c1 = F.relu(self.conv_norm1(c1))
            #
            l1 = self.lstm1(F.dropout(c1, ratio=self.dropout))
            l1 = self.lstm_norm1(l1)
            #
            c2 = self.conv2(F.dropout(l1, ratio=self.dropout))
            c2 = F.relu(self.conv_norm2(c2))
            #
            l2 = self.lstm2(F.dropout(c2, ratio=self.dropout))
            l2 = self.lstm_norm2(l2)
            #
            c3 = self.conv3(F.dropout(l2, ratio=self.dropout))
            c3 = F.relu(self.conv_norm3(c3))
            #
            l3 = self.lstm3(F.dropout(c3, ratio=self.dropout))
            l3 = self.lstm_norm3(l3)
            #
            c4 = self.conv4(F.dropout(l3, ratio=self.dropout))
            c4 = F.relu(self.conv_norm4(c4))
            #
            l4 = self.lstm4(F.dropout(c4, ratio=self.dropout))
            l4 = self.lstm_norm4(l4)
            #
            c5 = self.conv5(F.dropout(l4, ratio=self.dropout))
            c5 = F.relu(self.conv_norm5(c5))
            #
            l5 = self.lstm5(F.dropout(c5, ratio=self.dropout))
            l5 = self.lstm_norm5(l5)
            #
            cf = self.fc_conv(F.dropout(l5, ratio=self.dropout))
            lf = self.fc_lstm(F.dropout(cf, ratio=self.dropout))

        out = F.concat((self.fc_lstm.c, self.fc_lstm.h), axis=1)  # B,C,1,1 * 2 -> B,2C,1,1
        return out
