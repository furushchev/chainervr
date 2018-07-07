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


@click.group()
def cli():
    pass


@cli.command()
@click.argument("model_path")
@P.predict_params
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


@cli.command()
@click.argument("model_dir")
@P.predict_params
@click.option("--hidden-channels", type=int, default=1000)
@click.option("--num-episodes", type=int, default=5)
@click.option("--image-num", type=int, default=0)
def summary(model_path, gpu, out, split, hidden_channels,
            num_episodes, image_num):

    model = chainervr.models.DeepEpisodicMemory(
        hidden_channels=hidden_channels, num_episodes=num_episodes)

    model.reset_state()

    P.predict_summary(
        model=model,
        model_dir=model_dir,
        gpu=gpu,
        in_episodes=in_episodes,
        out_episodes=out_episodes,
        channels_num=3,
        out=out,
        split=split,
        image_num=image_num)


if __name__ == '__main__':
    cli()
