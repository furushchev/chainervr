#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer.functions as F
from chainer.backends import cuda


def gradient_difference_error(x, t, alpha=2.0):
    dxh = F.absolute(x[:,:,:,:,:-1] - x[:,:,:,:,1:])
    dth = F.absolute(t[:,:,:,:,:-1] - t[:,:,:,:,1:])
    dxv = F.absolute(x[:,:,:,:-1,:] - x[:,:,:,1:,:])
    dtv = F.absolute(t[:,:,:,:-1,:] - t[:,:,:,1:,:])

    dh = F.absolute(dxh - dth) ** alpha
    dv = F.absolute(dxv - dtv) ** alpha
    return (F.mean(dh) + F.mean(dv)) / 2.0
