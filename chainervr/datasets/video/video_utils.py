#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import imageio
import os
import os.path as osp


def check_video(vp):
    video = None
    try:
        video = imageio.get_reader(vp)
        meta = video.get_meta_data()
        return True
    except:
        return False
    finally:
        if video is not None:
            video.close()


def find_files(dir_path, ext, recursive=False, ignore_case=False):
    fns = []
    if recursive:
        for d in os.listdir(dir_path):
            fns += find_files(d, True)
    if ignore_case:
        fns += glob.glob(osp.join(dir_path, "*." + ext.lower()))
        fns += glob.glob(osp.join(dir_path, "*." + ext.upper()))
    else:
        fns += glob.glob(osp.join(dir_path, "*." + ext))
    return [osp.abspath(fn) for fn in fns]


def find_videos(dir_path, recursive=False):
    exts = ["mp4", "avi", "mov"]
    fns = []
    for ext in exts:
        fns += find_files(dir_path, ext, recursive=recursive)
    return fns

