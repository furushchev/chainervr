#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>


class SplitEpisode(object):
    def __init__(self, num_episodes):
        assert isinstance(num_episodes, list)
        self.num_episodes = num_episodes

    def __call__(self, x):
        """x: (chainer.Variable, ndarray)
              Array of dimension (N, D)
              where N is a number of frames of the episode.
        """

        assert x.shape[0] >= sum(self.num_episodes), "invalid input: %s" % str(x.shape)

        xs = []
        start = 0
        for length in self.num_episodes:
            end = start + length
            xs.append(x[start:end])
            start = end

        return tuple(xs)
