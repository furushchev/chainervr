#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import cv2
from collections import defaultdict
import chainer
import numpy as np
import os.path as osp
from chainercv import utils
import imageio
import pandas as pd
from epic_kitchen_utils import get_action_videos
from epic_kitchen_utils import parse_action_annotation
from epic_kitchen_action_labels import epic_kitchen_action_label_names


class EpicKitchenActionDataset(chainer.dataset.DatasetMixin):
    def __init__(self, split="train", data_dir="auto", anno_path="auto",
                 fps=1.0, resize=(128, 128),
                 force_download=False, download_timeout=None, use_cache=True,
                 skip_error_files=True, min_frames=None):
        if split not in ["train", "test"]:
            raise ValueError("Split '%s' not available" % split)

        if data_dir == "auto":
            if anno_path == "auto":
                data_dir, anno_path, annotations = get_action_videos(
                    split, force_download=force_download, download_timeout=download_timeout,
                    use_cache=use_cache, skip_error_files=skip_error_files, fps=fps, min_frames=min_frames)
            else:
                data_dir = get_action_videos(
                    split, force_download=force_download, download_timeout=download_timeout,
                    use_cache=use_cache, skip_error_files=skip_error_files, fps=fps, min_frames=min_frames)[0]
                annotations = parse_action_annotation(anno_path)
        else:
            if anno_path == "auto":
                raise ValueError("auto anno_path with specified data_dir is unsupported")
            else:
                annotations = parse_action_annotation(anno_path)

        self.data_dir = data_dir
        self.split = split
        self.annotations = annotations
        self.fps = fps
        self.resize = resize

    def __len__(self):
        return len(self.annotations)

    def get_video_path(self, annotation):
        pid, vid = annotation["participant_id"], annotation["video_id"]
        video_path = osp.join(self.data_dir, self.split, pid, vid + ".MP4")
        return video_path

    def get_images(self, annotation):
        video_path = self.get_video_path(annotation)
        video = imageio.get_reader(video_path)
        try:
            meta = video.get_meta_data()
            video_frames, video_fps = meta["nframes"], meta["fps"]
            nskips = int(video_fps // self.fps)

            start_frame = annotation["start_frame"]
            stop_frame = annotation["stop_frame"]

            images = []
            for i in range(start_frame, stop_frame, nskips):
                try:
                    img = video.get_data(i)
                    if self.resize:
                        img = cv2.resize(img, self.resize)
                except IndexError as e:
                    import traceback
                    print traceback.format_exc()
                    print meta
                    print i, start_frame, stop_frame, nskips
                    print range(start_frame, stop_frame, nskips)
                    raise e
                if img.ndim == 2:
                    img = img[np.newaxis]  # 1HW
                else:
                    img = img.transpose((2, 0, 1))  # CHW
                images.append(img)
            return np.asarray(images, dtype=np.float32)
        finally:
            video.close()

    def get_length(self, i):
        annotation = self.annotations[i]
        video_path = self.get_video_path(annotation)
        video = imageio.get_reader(video_path)
        try:
            meta = video.get_meta_data()
            video_frames, video_fps = meta["nframes"], meta["fps"]
            nskips = int(video_fps // self.fps)
            start_frame = annotation["start_frame"]
            stop_frame = annotation["stop_frame"]
            return len(range(start_frame, stop_frame, nskips))
        finally:
            video.close()

    def get_example(self, i):
        annotation = self.annotations[i]
        images = self.get_images(annotation)
        verb = annotation["verb_class"]
        nouns = annotation["all_noun_classes"]
        label = np.asarray([verb] + nouns, dtype=np.int32)
        return images, label


if __name__ == '__main__':
    from chainercv.visualizations import vis_bbox
    import matplotlib.pyplot as plt
    import traceback

    dataset = EpicKitchenActionDataset(split="train", fps=2.0)

    print "Loadded dataset. length:", len(dataset)
    print "Starting slideshow..."

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_init = False
    for i in range(0, len(dataset), 10):
        try:
            images, label = dataset.get_example(i)
        except Exception as e:
            print i, e
            traceback.print_exc()
        frames, channels, height, width = images.shape
        dummy_bbox = np.asarray([[0, 0, height, width]])
        for i in range(frames):
            img = images[i]
            ax = vis_bbox(img, dummy_bbox, [label[0]], label_names=epic_kitchen_action_label_names, ax=ax)
            if plot_init is False:
                plt.show()
                plot_init = True
            else:
                fig.canvas.draw()
                fig.canvas.flush_events()
            plt.pause(0.1)
            ax.clear()
