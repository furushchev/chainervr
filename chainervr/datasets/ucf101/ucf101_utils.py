#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
from collections import OrderedDict
import functools
import glob
import imageio
import math
import multiprocessing as mp
import numpy as np
import os
import pandas as pd
from .. import utils


root = "furushchev/chainervr/ucf101"
video_url = "http://crcv.ucf.edu/data/UCF101/UCF101.rar"
anno_url = "http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
fps = 4.0


def apply(func, prop):
    try:
        if isinstance(prop, list) or isinstance(prop, tuple):
            return func(*prop)
        elif isinstance(prop, dict):
            return func(**prop)
        else:
            return func(prop)
    except Exception as err:
        return err


def extract_episode(src, dst, min_frames=None, force=False):
    """Extract images from video. Returns True if successfully extracted, False otherwise"""

    if not force and os.path.exists(dst):
        images = glob.glob(os.path.join(dst, "frame_*.jpg"))
        if min_frames is not None and len(images) < min_frames:
            raise IndexError("short length of video %s: %d" % (src, len(images)))
        return True

    v = None
    try:
        try:
            v = imageio.get_reader(src)
        except imageio.core.fetching.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            v = imageio.get_reader(src)
        meta = v.get_meta_data()
        src_nframes, src_fps = meta["nframes"], meta["fps"]
        step = math.floor(src_fps // fps)
        indices = range(0, src_nframes, step)
        os.makedirs(dst, exist_ok=True)
        i = 0
        for index in indices:
            for offset in [0, -1, 1]:
                try:
                    img = v.get_data(index + offset)
                    fn = "frame_%04d.jpg" % i
                    imageio.imwrite(os.path.join(dst, fn), img)
                    break
                except Exception as e:
                    continue
            i += 1
        if min_frames is not None and i < min_frames:
            raise IndexError("short length of video %s: %d" % (src, i))
        return True
    except Exception as e:
        raise e
    finally:
        if v is not None:
            v.close()


def resolve_video_path(name):
    category, video = os.path.split(name)
    video_base, video_ext = os.path.splitext(video)
    base_dir = os.path.join(utils.get_dataset_directory("ucf101"), "UCF-101")
    video_path = os.path.join(base_dir, name)
    images_dir = os.path.join(base_dir, category, video_base)
    return video_path, images_dir


def load_episode(name, num_episodes=None, skip=None):
    import imageio

    if num_episodes is None:
        num_episodes = 5
    if skip is None:
        skip = 1

    video_path, images_dir = resolve_video_path(name)

    if not os.path.exists(images_dir):
        print("Video '%s' is not yet pre-processed." % name)
        ok = extract_episode(video_path, images_dir)
        assert ok, "failed to extract episode"

    all_image_names = sorted(glob.glob(os.path.join(images_dir, "frame_*.jpg")))
    image_names = []
    indices = range(0, num_episodes * skip, skip)
    if max(indices) > len(all_image_names):
        raise IndexError("list index out of range: %s" % name)
    for i in indices:
        image_names.append(all_image_names[i])
    images = []
    for image_name in image_names:
        img = imageio.imread(image_name).astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))  # HWC -> CHW
        images.append(img)
    return np.asarray(images)


def get_ucf101(split, nepisodes=None, npattern=None, nprocs=None):
    # download annotation
    if nepisodes is None:
        nepisodes = 10
    if npattern is None:
        npattern = 1
    elif npattern not in [1, 2, 3]:
        raise ValueError("npattern must be 1, 2 or 3")
    if split == "train":
        anno_path = os.path.join("ucfTrainTestlist", "trainlist%02d.txt" % npattern)
    elif split == "test":
        anno_path = os.path.join("ucfTrainTestlist", "testlist%02d.txt" % npattern)
    anno = utils.cache_or_load_file(
        "ucf101", anno_url, anno_path,
        lambda p: pd.read_csv(p, sep=" ", names=["filename", "class"]),
    )

    # download videos
    video_root_dir = utils.cache_or_load_file(
        "ucf101", video_url, "UCF-101",
        lambda p: p
    )

    # load or extract images
    if nprocs is None:
        nprocs = mp.cpu_count() // 4
    pool = mp.Pool(nprocs)
    props = [resolve_video_path(name) + (nepisodes,) for name in anno["filename"]]
    results = pool.map_async(functools.partial(apply, extract_episode), props).get(600)

    drop_indices = []
    for i, result in enumerate(results):
        if isinstance(result, IndexError):
            drop_indices.append(i)
        elif isinstance(result, Exception):
            name = anno["filename"][i]
            raise RuntimeError("Failed to extract video %s: %s" % (name, result))

    anno = anno.drop(anno.index[drop_indices])

    if len(drop_indices) > 0:
        print("%d of %d videos are dropped due to short of length: %d" % (len(drop_indices), len(results), anno.shape[0]))

    return {"root_dir": video_root_dir,
            "annotations": anno,}


ucf101_class_names = (
    "None",
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BreastStroke",
    "BrushingTeeth",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "Diving",
    "Drumming",
    "Fencing",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "FrontCrawl",
    "GolfSwing",
    "Haircut",
    "Hammering",
    "HammerThrow",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "HulaHoop",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpingJack",
    "JumpRope",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MilitaryParade",
    "Mixing",
    "MoppingFloor",
    "Nunchucks",
    "ParallelBars",
    "PizzaTossing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "SkateBoarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo",
)


if __name__ == '__main__':
    print(get_ucf101("train"))
