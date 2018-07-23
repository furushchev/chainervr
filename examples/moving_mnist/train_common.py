#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("Agg")
import multiprocessing as mp
import click
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


def train(model, train_chain, channels_num, in_episodes, out_episodes,
          gpu, multi_gpu, out,
          batch_size, max_iter, resume,
          log_interval, snapshot_interval):

    comm = None
    if multi_gpu:
        if not multi_gpu_available:
            raise click.BadParameter("chainermn is not yet installed")
        info("Using multiple GPU")
        comm = chainermn.create_communicator(communicator_name="pure_nccl")
        gpu = comm.intra_rank

    if gpu >= 0:
        info("Training with GPU %d" % gpu)
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    else:
        info("Training with CPU")

    info("Loading dataset")
    if not multi_gpu or comm.rank == 0:
        train_data = chainer.datasets.TransformDataset(
            chainervr.datasets.MovingMnistDataset(
                split="train", channels_num=channels_num),
            chainervr.datasets.SplitEpisode([in_episodes, out_episodes]))
        test_data = chainer.datasets.TransformDataset(
            chainervr.datasets.MovingMnistDataset(
                split="test", channels_num=channels_num),
            chainervr.datasets.SplitEpisode([in_episodes, out_episodes]))
    else:
        train_data = test_data = None
    nprocs = None
    if multi_gpu:
        try:
            mp.set_start_method("forkserver")
        except:
            info("MultiprocessIterator may crash on multi-GPU mode")
        nprocs = mp.cpu_count() // comm.size // 2
        info("Using %d procs" % nprocs)
        train_data = chainermn.scatter_dataset(
            train_data, comm, shuffle=True)
        test_data = chainermn.scatter_dataset(
            test_data, comm, shuffle=True)
    train_iter = chainer.iterators.MultiprocessIterator(
        train_data, batch_size, repeat=True, shuffle=True,
        n_processes=nprocs)
    test_iter = chainer.iterators.SerialIterator(
        test_data, batch_size, repeat=False, shuffle=False)

    info("Setting trainer")
    optimizer = chainer.optimizers.Adam()
    if multi_gpu:
        optimizer = chainermn.create_multi_node_optimizer(
            optimizer, comm)
    optimizer.setup(train_chain)

    updater = chainer.training.StandardUpdater(
        train_iter, optimizer, device=gpu)
    trainer = training.Trainer(
        updater, (max_iter, "iteration"), out=out)

    evaluator = extensions.Evaluator(test_iter, train_chain, device=gpu)
    if multi_gpu:
        evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator)

    if not multi_gpu or comm.rank == 0:
        trainer.extend(extensions.LogReport(trigger=(log_interval, "iteration")))
        trainer.extend(extensions.observe_lr(), trigger=(log_interval, "iteration"))
        trainer.extend(extensions.PrintReport(
            ["epoch", "iteration", "lr",
             "main/loss", "main/loss/reconst", "main/loss/pred",
             "validation/main/loss"]),
                       trigger=(log_interval, "iteration"))
        trainer.extend(extensions.PlotReport(
            ["main/loss", "main/loss/reconst", "main/loss/pred",
             "validation/main/loss"]),
                       trigger=(log_interval, "iteration"))
        trainer.extend(extensions.ProgressBar(update_interval=1))
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

