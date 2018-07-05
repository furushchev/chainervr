#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
import os.path as osp
import sys
import chainervr

_THIS_DIR = osp.dirname(osp.abspath(__file__))
sys.path.insert(0, osp.join(_THIS_DIR, ".."))

import predict_common as P


@click.command()
@click.argument("model_path")
@click.option("--gpu", type=int, default=-1)
@click.option("--out", type=str, default="predicts")
@click.option("--split", type=str, default="test")
@click.option("--hidden-channels", type=int, default=1000)
@click.option("--num-episodes", type=int, default=5)
@click.option("--start-from", type=int, default=0)
@click.option("--images-num", type=int, default=100)
def predict(model_path, gpu, out, split, hidden_channels,
            num_episodes, start_from, images_num):

    P.info("Loading model from %s" % model_path)

    model = chainervr.models.DeepEpisodicMemory(
        hidden_channels=hidden_channels, num_episodes=num_episodes)

    model.reset_state()

    P.predict(
        model=model,
        model_path=model_path,
        gpu=gpu, out=out,
        in_episodes=num_episodes,
        out_episodes=num_episodes,
        channels_num=3,
        split=split,
        start_from=start_from,
        images_num=images_num)



if __name__ == '__main__':
    predict()
