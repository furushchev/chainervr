#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import chainervr
from chainervr.visualizations import vis_episode


@click.command()
@click.argument("index", type=int)
@click.option("--split", type=str, default="train")
@click.option("--vertical", is_flag=True)
@click.option("--out", "-o", type=str, default="output.png")
def show_dataset(index, split, vertical, out):
    dataset = chainervr.datasets.MovingMnistDataset(
        split=split, channels_num=1,
    )

    in_data = dataset[index]

    ax = vis_episode(in_data, vertical=vertical)
    plt.title("index %s" % index)
    plt.savefig(out, bbox_inches="tight", dpi=300)
    print("Saved to %s" % out)


if __name__ == '__main__':
    show_dataset()
