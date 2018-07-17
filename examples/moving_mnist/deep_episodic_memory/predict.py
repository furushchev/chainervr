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
@click.option("--start-from", type=int, default=0)
@click.option("--images-num", type=int, default=100)
def predict(model_path, gpu, out, split, hidden_channels,
            in_episodes, out_episodes, start_from, images_num):
    """Generate image for visualization using trained model"""
    assert in_episodes == out_episodes

    P.info("Loading model from %s" % model_path)

    model = chainervr.models.DeepEpisodicMemory(
        hidden_channels=hidden_channels, num_episodes=in_episodes)

    model.reset_state()

    P.predict(
        model=model,
        model_path=model_path,
        gpu=gpu, out=out,
        in_episodes=in_episodes,
        out_episodes=out_episodes,
        channels_num=3,
        split=split,
        start_from=start_from,
        images_num=images_num)


@cli.command()
@click.argument("model_dir")
@P.predict_params
@click.option("--hidden-channels", type=int, default=1000)
@click.option("--image-num", type=int, default=0)
def summary(model_dir, gpu, out, split, hidden_channels,
            in_episodes, out_episodes, image_num):
    """Generate images for evaluation using each model in directory"""
    assert in_episodes == out_episodes

    model = chainervr.models.DeepEpisodicMemory(
        hidden_channels=hidden_channels, num_episodes=in_episodes)

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


@cli.command()
@P.predict_params
@click.argument("model_path")
@click.option("--hidden-channels", type=int, default=1000)
@click.option("--start-from", type=int, default=0)
@click.option("--images_num", type=int, default=None)
def extract(model_path, gpu, out, split, in_episodes, out_episodes,
            hidden_channels, start_from, images_num):
    """Extract hidden features for each data in dataset"""
    import chainer
    import chainervr
    import os
    import numpy as np
    import pandas as pd

    assert in_episodes == out_episodes

    model = chainervr.models.DeepEpisodicMemory(
        hidden_channels=hidden_channels, num_episodes=in_episodes)

    model.reset_state()

    if gpu >= 0:
        P.info("Using GPU %d" % gpu)
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    else:
        P.info("Using CPU")

    P.info("Loading dataset")

    dataset = chainervr.datasets.MovingMnistDataset(
        split=split, channels_num=3)

    os.makedirs(out, exist_ok=True)

    chainer.config.train = False

    P.info("Forwarding")

    if images_num is None:
        images_num = len(dataset)

    xp = model.xp
    features = []
    with click.progressbar(range(start_from, images_num), label="Extracting") as it:
        for n in it:
            data = dataset[n]
            in_data, next_data = data[:in_episodes], data[in_episodes:in_episodes+out_episodes]
            in_data, next_data = in_data[np.newaxis, :], next_data[np.newaxis, :]

            in_data = chainer.Variable(in_data)
            if gpu >= 0:
                with chainer.cuda.get_device_from_id(gpu):
                    in_data.to_gpu()

            _, _, hidden = model(in_data)
            if gpu >= 0:
                hidden.to_cpu()
            feature = hidden.array.reshape(-1)
            features.append(feature)

    features = np.asarray(features, dtype=features[0].dtype)
    save_fn = os.path.join(out, "features.npz")
    np.savez(save_fn, features)
    P.info("Extracted features into %s" % save_fn)


if __name__ == '__main__':
    cli()
