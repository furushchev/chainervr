#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
import os.path as osp
import sys
import chainervr
import chainervr.utils.train as T


@click.command()
@click.option("--batch-size", type=int, default=8)
@click.option("--max-iter", type=int, default=100000)
@click.option("--gpu", type=int, default=-1)
@click.option("--multi-gpu", is_flag=True)
@click.option("--disable-predict", is_flag=True)
@click.option("--out", type=str, default="results")
@click.option("--in-episodes", type=int, default=5)
@click.option("--out-episodes", type=int, default=5)
@click.option("--log-interval", type=int, default=100)
@click.option("--snapshot-interval", type=int, default=100)
@click.option("--resume", type=str, default="")
def train(batch_size, max_iter,
          gpu, multi_gpu, out,
          disable_predict,
          in_episodes, out_episodes,
          log_interval, snapshot_interval, resume):
    T.info("Loading model")

    model = chainervr.models.ConvLSTM(
        n_channels=1, patch_size=(64, 64),
        in_episodes=in_episodes, out_episodes=out_episodes,
        predict=not disable_predict)
    train_chain = chainervr.models.EpisodicTrainChain(
        model, ratio=0.5)

    model.reset_state()

    T.train(
        model=model,
        train_chain=train_chain,
        train_dataset_cls=chainervr.datasets.MovingMnistDataset,
        test_dataset_cls=chainervr.datasets.MovingMnistDataset,
        dataset_args={"channels_num": 1},
        in_episodes=in_episodes,
        out_episodes=out_episodes,
        gpu=gpu, multi_gpu=multi_gpu, out=out,
        batch_size=batch_size, max_iter=max_iter,
        resume=resume,
        log_interval=log_interval,
        snapshot_interval=snapshot_interval,
    )


if __name__ == '__main__':
    train()
