#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os
import click
import pandas as pd
if not os.getenv("DISPLAY", None):
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


@click.command()
@click.argument("log_path")
@click.option("--out", type=str, default="plot.png")
def plot(log_path, out):
    df = pd.read_json(log_path)
    df.plot(x="iteration", y=["main/loss"])
    plt.savefig(out)
    plt.show()


if __name__ == '__main__':
    plot()
