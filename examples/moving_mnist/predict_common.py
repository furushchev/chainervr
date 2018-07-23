#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import click
import copy
import functools
import glob
import numpy as np

import chainer
import chainer.functions as F

import chainervr
from chainervr.visualizations import vis_episode


def info(msg):
    click.secho(msg, fg="green")


def predict_params(func):
    @click.option("--gpu", type=int, default=-1)
    @click.option("--out", type=str, default="predicts")
    @click.option("--split", type=str, default="test")
    @click.option("--in-episodes", type=int, default=5)
    @click.option("--out-episodes", type=int, default=5)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def predict(model, model_path, gpu,
            in_episodes, out_episodes, channels_num,
            out, split, start_from, images_num):
    if gpu >= 0:
        info("Using GPU %d" % gpu)
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    else:
        info("Using CPU")

    chainer.serializers.load_npz(model_path, model)

    info("Loading dataset")

    dataset = chainervr.datasets.MovingMnistDataset(split=split, channels_num=channels_num)

    os.makedirs(out, exist_ok=True)

    chainer.config.train = False

    info("Forwarding")

    xp = model.xp
    for n in range(start_from, images_num):
        data = dataset[n]
        in_data, next_data = data[:in_episodes], data[in_episodes:in_episodes+out_episodes]
        in_data, next_data = in_data[np.newaxis, :], next_data[np.newaxis, :]

        in_data = chainer.Variable(in_data)
        if gpu >= 0:
            with chainer.cuda.get_device_from_id(gpu):
                in_data.to_gpu()

        out_data = model(in_data)
        reconst, pred = out_data[0], out_data[1]

        fig = plt.figure()

        ax = fig.add_subplot(2, 2, 1)
        ax = vis_episode(in_data, ax=ax, title="Input")

        ax = fig.add_subplot(2, 2, 2)
        ax = vis_episode(next_data, ax=ax, title="Predicted Ground Truth")

        ax = fig.add_subplot(2, 2, 3)
        ax = vis_episode(reconst, ax=ax, title="Reconstruction")

        ax = fig.add_subplot(2, 2, 4)
        ax = vis_episode(pred, ax=ax, title="Prediction")

        out_path = os.path.join(out, "result_%03d.png" % n)
        plt.savefig(out_path, bbox_inches="tight", dpi=160)
        info("saved to %s" % out_path)
        plt.close(fig)


def predict_summary(model, model_dir, gpu,
                    in_episodes, out_episodes, channels_num,
                    out, split, image_num):
    if gpu >= 0:
        info("Using GPU %d" % gpu)
        chainer.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    else:
        info("Using CPU")

    glob_ex = os.path.join(model_dir, "model_iter_")
    model_paths = glob.glob(glob_ex + "*")
    model_paths.sort(key=lambda s: int(s[len(glob_ex):]))
    info("Found %d models" % len(model_paths))

    info("Loading dataset")

    dataset = chainervr.datasets.MovingMnistDataset(split=split, channels_num=channels_num)

    data = dataset[image_num]
    in_data, next_data = data[:in_episodes], data[in_episodes:in_episodes+out_episodes]
    in_data, next_data = in_data[np.newaxis, :], next_data[np.newaxis, :]

    in_data = chainer.Variable(in_data)
    if gpu >= 0:
        with chainer.cuda.get_device_from_id(gpu):
            in_data.to_gpu()


    rows = len(model_paths) + 1
    fig = plt.figure(figsize=(3, rows))
    i = 1

    ax = fig.add_subplot(rows, 2, i)
    ax = vis_episode(in_data, ax=ax, title="Input")
    i += 1

    ax = fig.add_subplot(rows, 2, i)
    ax = vis_episode(in_data, ax=ax, title="Predicted Ground Truth")
    i += 1

    for model_path in model_paths:
        model_name = os.path.basename(model_path)
        chainer.serializers.load_npz(model_path, model)
        chainer.config.train = False

        out_data = model(in_data)
        reconst, pred = out_data[0], out_data[1]

        ax = fig.add_subplot(rows, 2, i)
        ax = vis_episode(reconst, ax=ax, title="%s Reconst" % model_name)
        i += 1

        ax = fig.add_subplot(rows, 2, i)
        ax = vis_episode(pred, ax=ax, title="%s Pred" % model_name)
        i += 1
        info("Generated image using %s" % model_path)

    os.makedirs(out, exist_ok=True)
    out_path = os.path.join(out, "summary.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    info("saved to %s" % out_path)
    plt.close(fig)
