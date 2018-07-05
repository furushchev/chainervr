#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("Agg")

import click
import numpy as np
import multiprocessing as mp

import chainer
from chainer import serializers
from chainer import training
from chainer.training import extensions
try:
    import chainermn
    multi_gpu_available = True
except:
    multi_gpu_available = False

import chainervr


def info(msg):
    click.secho(msg, fg="green")


@click.command()
@click.option("--batch-size", type=int, default=20)
@click.option("--max-iter", type=int, default=100000)
@click.option("--gpu", type=int, default=-1)
@click.option("--multi-gpu", is_flag=True)
@click.option("--out", type=str, default="results")
@click.option("--episode-size", type=int, default=10)
@click.option("--loss-func", type=click.Choice(["mse", "gdl", "mse_gdl"]), default="mse_gdl")
@click.option("--log-interval", type=int, default=10)
@click.option("--snapshot-interval", type=int, default=100)
@click.option("--resume", type=str, default="")
def train(batch_size, max_iter, loss_func,
          gpu, multi_gpu, out, episode_size,
          log_interval, snapshot_interval, resume):

    info("Preparing model")
    model = chainervr.models.DeepEpisodicMemory(
        hidden_channels=1000, out_channels=3,
        episode_size=episode_size)
    train_chain = chainervr.models.DeepEpisodicMemoryTrainChain(
        model, loss_func=loss_func)

    train_chain.reset_state()

    multi_gpu_comm = None
    if multi_gpu:
        if not multi_gpu_available:
            raise click.BadParameter("chainermn is not yet installed")
        info("Using multi GPU")
        multi_gpu_comm = chainermn.create_communicator()
        gpu = multi_gpu_comm.intra_rank

    if gpu >= 0:
        info("Training with GPU %d" % gpu)
        model.to_gpu(gpu)
    else:
        info("Training with CPU mode")

    info("Loading dataset")
    if multi_gpu:
        if multi_gpu_comm.rank == 0:
            train_data = chainervr.datasets.MovingMnistDataset()
        else:
            train_data = None
        train_data = chainermn.scatter_dataset(
            train_data, multi_gpu_comm, shuffle=True)
        if hasattr(mp, "set_start_method"):
            # to avoid crash
            mp.set_start_method("forkserver")
            train_iter = chainer.iterators.MultiprocessIterator(
                train_data, batch_size,
                n_processes=mp.cpu_count() // multi_gpu_comm.size)
    else:
        train_data = chainervr.datasets.MovingMnistDataset()
        train_iter = chainer.iterators.MultiprocessIterator(
            train_data, batch_size)

    info("Configuring trainer")
    optimizer = chainer.optimizers.Adam()
    if multi_gpu:
        optimizer = chainermn.create_multi_node_optimizer(
            optimizer, multi_gpu_comm)
    optimizer.setup(train_chain)

    updater = chainervr.training.EpisodicUpdater(train_iter, optimizer, device=gpu)
    trainer = training.Trainer(updater, (max_iter, "iteration"), out=out)

    if not multi_gpu or multi_gpu_comm.rank == 0:
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
