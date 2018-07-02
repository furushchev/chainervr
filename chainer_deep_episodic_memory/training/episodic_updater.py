#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer
from chainer import training


class EpisodicUpdater(training.updaters.StandardUpdater):
    def __init__(self, train_iter, optimizer, device):
        super(EpisodicUpdater, self).__init__(
            train_iter, optimizer, device=device,)

    def update_core(self):
        train_iter = self.get_iterator("main")
        optimizer = self.get_optimizer("main")
        model = optimizer.target

        loss = 0

        in_data = next(train_iter)  # B x (NiCHW, label)
        xs = [d[0] for d in in_data]  # use images only
        dtype = xs[0].dtype
        x = np.asarray([x[:model.episode_size+1] for x in xs], dtype=dtype)  # Ni -> N
        model.reset_state()
        x = x.transpose((1, 0, 2, 3, 4))  # NBCHW
        episode_size = x.shape[0] - 1
        xi = None
        for i in range(episode_size):
            if xi is None:
                xi = chainer.dataset.to_device(self.device, x[i])
            pi = chainer.dataset.to_device(self.device, x[i+1])
            ri = xi
            loss += model(chainer.Variable(xi),
                          chainer.Variable(ri),
                          chainer.Variable(pi))
            xi = pi

        model.cleargrads()
        loss.backward()
        loss.unchain_backward()
        optimizer.update()
