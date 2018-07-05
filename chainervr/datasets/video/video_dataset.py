#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import os.path as osp
import imageio
import numpy as np
import chainer
from .video_utils import find_videos
from .video_utils import check_video


class VideoDataset(chainer.dataset.DatasetMixin):
    def __init__(self, root_dir, recursive=False, fps=1.0, resize=(128, 128)):
        if not osp.exists(root_dir):
            raise IOError("%s does not exist" % root_dir)


        video_paths = find_videos(root_dir, recursive=recursive)
        self.video_paths = filter(check_video, video_paths)
        self.fps = fps
        self.resize = resize

    def __len__(self):
        return len(self.video_paths)

    def get_example(self, i):
        video_path = self.video_paths[i]
        video = None
        images = []
        try:
            video = imageio.get_reader(video_path)
            meta = video.get_meta_data()
            nframes, fps = meta["nframes"], meta["fps"]
            nskips = int(fps // self.fps)
            for frame in range(0, nframes, nskips):
                try:
                    img = video.get_data(frame)
                    if self.resize:
                        img = cv2.resize(img, self.resize)
                except IndexError as e:
                    raise e
                if img.ndim == 2:
                    img = img[np.newaxis]  # 1HW
                else:
                    img = img.transpose((2, 0, 1))  # CHW
                images.append(img)
        finally:
            if video is not None:
                video.close()
        return np.asarray(images, dtype=np.float32)
