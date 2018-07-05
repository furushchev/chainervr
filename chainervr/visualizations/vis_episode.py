#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>


def vis_episode(in_data, ax=None, vertical=False, anti_aliasing=False):
    """Visualize image sequence

    Args:
        in_data (list of ndarray (CHW) or ndarray (NCHW)): input image sequence
        ax (matplotlib.axes.Axis): Axis on which images are drawn. A new axis is created if this value is `None`.
        vertical (bool): Images are drawn vertically if this value is `True`, otherwise drawn horizontally.
        anti_aliasing (bool): Resize with anti-aliasing if this value is `True`.
    Returns:
        Returns the axis where images are drawn.
    """

    import matplotlib.pyplot as plt
    import numpy as np
    from skimage.transform import rescale

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    nframes = None
    if isinstance(in_data, list):
        # resize
        idx = 2 if vertical else 1
        target_size = max(map(lambda a: a.shape[idx], in_data))
        for i, img in enumerate(in_data):
            # CHW -> HWC
            size = img.shape[idx]
            img = img.transpose((1, 2, 0))
            scale = 1.0 * target_size / size
            img = rescale(img, scale, mode="reflect", multichannel=True, anti_aliasing=anti_aliasing)
            in_data[i] = img
        in_data = np.concatenate(in_data, axis=0)
    else:
        # NCHW -> NHWC
        in_data = in_data.transpose((0, 2, 3, 1))

    # stack
    if vertical:
        in_data = in_data.reshape(-1, *in_data.shape[2:])  # HWC
    else:
        in_data = in_data.transpose((0, 2, 1, 3))  # NWHC
        in_data = in_data.reshape(-1, *in_data.shape[2:])
        in_data = in_data.transpose((1, 0, 2))  # HWC

    channels_num = in_data.shape[-1]
    if channels_num == 1:
        in_data = np.broadcast_to(in_data, in_data.shape[:-1] + (3,))

    ax.imshow(in_data)

    return ax
