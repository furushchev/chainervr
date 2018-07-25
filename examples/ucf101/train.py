#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
import os.path as osp
import sys
import chainer.functions as F
import chainervr
import chainervr.utils.train as T


def mse_gd_loss(x, t, eta=0.5):
    mse = F.mean_squared_error(x, t)
    gd = chainervr.functions.gradient_difference_error(x, t)
    return mse * (1.0 - eta) + gd * eta


@click.command()
@click.option("--batch-size", type=int, default=16)
@click.option("--max-iter", type=int, default=100000)
@click.option("--gpu", type=int, default=-1)
@click.option("--multi-gpu", is_flag=True)
@click.option("--out", type=str, default="results")
@click.option("--loss-func", type=click.Choice(["mse", "gd", "mse_gd"]), default="mse_gd")
@click.option("--eta", type=float, default=0.4)
@click.option("--hidden-channels", type=int, default=1000)
@click.option("--dropout", type=float, default=0.1)
@click.option("--noise-sigma", type=float, default=0.1)
@click.option("--num-episodes", type=int, default=5)
@click.option("--log-interval", type=int, default=10)
@click.option("--snapshot-interval", type=int, default=1000)
@click.option("--resume", type=str, default="")
def train(batch_size, max_iter,
          gpu, multi_gpu, out,
          loss_func, eta,
          hidden_channels, num_episodes,
          dropout, noise_sigma,
          log_interval, snapshot_interval, resume):

    T.info("Loading model")

    model = chainervr.models.DeepEpisodicMemory(
        hidden_channels=hidden_channels, num_episodes=num_episodes,
        dropout=dropout, noise_sigma=noise_sigma)

    T.info("Using %s as loss function" % loss_func)
    if loss_func == "mse":
        loss_func = F.mean_squared_error
    elif loss_func == "gd":
        loss_func = chainervr.functions.gradient_difference
    else:
        loss_func = lambda x, t: mse_gd_loss(x,t, eta)

    train_chain = chainervr.models.EpisodicTrainChain(
        model, ratio=0.5, loss_func=loss_func)

    model.reset_state()

    T.info("Loading dataset")

    train_dataset = chainervr.datasets.UCF101Dataset(
        split="train", num_episodes=num_episodes*2)
    test_dataset = chainervr.datasets.UCF101Dataset(
        split="test", num_episodes=num_episodes*2)

    T.train(
        model=model,
        train_chain=train_chain,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        in_episodes=num_episodes,
        out_episodes=num_episodes,
        gpu=gpu, multi_gpu=multi_gpu, out=out,
        batch_size=batch_size, max_iter=max_iter,
        resume=resume,
        log_interval=log_interval,
        snapshot_interval=snapshot_interval,
    )


if __name__ == '__main__':
    train()
