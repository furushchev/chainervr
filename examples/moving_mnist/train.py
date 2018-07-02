#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("Agg")

import click
import numpy as np

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions

import chainer_deep_episodic_memory as D


def info(msg):
    click.secho(msg, fg="green")


@click.command()
@click.option("--batch-size", type=int, default=20)
@click.option("--max-iter", type=int, default=100000)
@click.option("--gpu", type=int, default=0)
@click.option("--out", type=str, default="results")
@click.option("--episode-size", type=int, default=10)
@click.option("--loss-func", type=click.Choice(["mse", "gdl", "mse_gdl"]), default="mse_gdl")
@click.option("--log-interval", type=int, default=10)
@click.option("--snapshot-interval", type=int, default=100)
@click.option("--resume", type=str, default="")
def train(batch_size, max_iter, loss_func,
          gpu, out, episode_size,
          log_interval, snapshot_interval, resume):

    info("Preparing model")
    model = D.models.DeepEpisodicMemory(
        hidden_channels=1000, out_channels=3,
        episode_size=episode_size)
    train_chain = D.models.DeepEpisodicMemoryTrainChain(
        model, loss_func=loss_func)

    train_chain.reset_state()
    if gpu >= 0:
        model.to_gpu(gpu)

    info("Loading dataset")
    train_data = D.datasets.MovingMnistDataset()
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size)

    info("Configuring trainer")
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(train_chain)

    updater = D.training.EpisodicUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (max_iter, "iteration"), out=out)

    trainer.extend(extensions.LogReport(trigger=(log_interval, "iteration")))
    trainer.extend(extensions.observe_lr(), trigger=(log_interval, "iteration"))
    trainer.extend(extensions.PrintReport(
        ["epoch", "iteration", "lr",
         "main/loss", "main/loss/mse", "main/loss/gdl"]),
                   trigger=(log_interval, "iteration"))
    trainer.extend(extensions.PlotReport(
        ["main/loss", "main/loss/mse", "main/loss/gdl"]),
                   trigger=(log_interval, "iteration"))
    trainer.extend(extensions.ProgressBar(update_interval=log_interval))
    trainer.extend(extensions.dump_graph(
        root_name="main/loss", out_name="network.dot"))
    trainer.extend(extensions.snapshot(),
                   trigger=(snapshot_interval, "iteration"))
    trainer.extend(extensions.snapshot_object(
        model, "model_iter_{.updater.iteration}"),
                   trigger=(snapshot_interval, "iteration"))

    if resume:
        info("Resume from %s" % resume)
        serializers.load_npz(resume, trainer)

    info("Start training")

    trainer.run()

    info("Done")


if __name__ == '__main__':
    train()
