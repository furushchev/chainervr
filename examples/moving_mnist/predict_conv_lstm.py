#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import click
import numpy as np
import multiprocessing as mp

import chainer

import chainervr
from chainervr.visualizations import vis_episode


def info(msg):
    click.secho(msg, fg="green")


def extract(in_data, gpu):
    if isinstance(in_data, chainer.Variable):
        if gpu >= 0:
            in_data.to_cpu()
        in_data = in_data.array
    in_data = in_data[0]  # NCHW
    channels = in_data.shape[1]
    if channels == 1:
        out_shape = list(in_data.shape)
        out_shape[1] = 3
        in_data = np.broadcast_to(in_data.copy(), out_shape)
    # lower bound = 0
    in_data.flags.writeable = True
    in_data[in_data <= 0.0] = 0.0
    return in_data


@click.command()
@click.argument("model_path")
@click.option("--gpu", "-g", type=int, default=-1)
@click.option("--out", "-o", type=str, default="conv_lstm_predict")
@click.option("--split", type=str, default="test")
@click.option("--disable-predict", is_flag=True)
@click.option("--in-episode", type=int, default=5)
@click.option("--out-episode", type=int, default=5)
def predict(model_path, gpu, out, split, disable_predict, in_episode, out_episode):

    info("Loading model from %s" % model_path)

    model = chainervr.models.ConvLSTM(
        n_channels=1, patch_size=(64, 64),
        predict=not disable_predict,
        in_episodes=in_episode, out_episodes=out_episode)

    model.reset_state()
    if gpu >= 0:
        info("Using GPU %d" % gpu)
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    else:
        info("Using CPU")

    chainer.serializers.load_npz(model_path, model)

    info("Loading dataset")

    dataset = chainervr.datasets.MovingMnistDataset(split=split, channels_num=1)

    os.makedirs(out, exist_ok=True)

    info("Forwarding")

    xp = model.xp
    for n in range(100):
        data = dataset[n]
        in_data, next_data = data[:in_episode], data[in_episode:in_episode+out_episode]
        in_data, next_data = in_data[np.newaxis, :], next_data[np.newaxis, :]

        with chainer.cuda.get_device_from_id(gpu):
            in_data = chainer.Variable(in_data)
            in_data.to_gpu()

        reconst, pred = model(in_data)

        in_data = extract(in_data, gpu)
        next_data = extract(next_data, gpu)
        reconst = extract(reconst, gpu)
        pred = extract(pred, gpu)

        fig = plt.figure(dpi=160)

        ax = fig.add_subplot(2, 2, 1)
        ax = vis_episode(in_data, ax=ax)
        ax.set_title("Input")
        ax.set_axis_off()

        ax = fig.add_subplot(2, 2, 2)
        ax = vis_episode(next_data, ax=ax)
        ax.set_title("Predict Ground Truth")
        ax.set_axis_off()

        ax = fig.add_subplot(2, 2, 3)
        ax = vis_episode(reconst, ax=ax)
        ax.set_title("Reconstruction")
        ax.set_axis_off()

        ax = fig.add_subplot(2, 2, 4)
        ax = vis_episode(pred, ax=ax)
        ax.set_title("Prediction")
        ax.set_axis_off()

        out_path = os.path.join(out, "result_%03d.png" % n)
        plt.savefig(out_path, bbox_inches="tight")
        info("saved to %s" % out_path)
        plt.close(fig)


if __name__ == '__main__':
    predict()
