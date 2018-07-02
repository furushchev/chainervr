#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import chainer.functions as F


def gradient_difference(x, t, alpha=2.0):
    dx_h = F.absolute_error(x[:,:,:,:-1], x[:,:,:,1:])
    dt_h = F.absolute_error(t[:,:,:,:-1], t[:,:,:,1:])
    dx_v = F.absolute_error(x[:,:,:-1,:], x[:,:,1:,:])
    dt_v = F.absolute_error(t[:,:,:-1,:], t[:,:,1:,:])

    dh = F.absolute_error(dx_h, dt_h) ** alpha
    dv = F.absolute_error(dx_v, dt_v) ** alpha

    return (F.mean(dh) + F.mean(dv)) / 2.0
