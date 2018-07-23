#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import click
from collections import OrderedDict
import glob
import imageio
import math
import numpy as np
import os
import pandas as pd
from .. import utils


root = "furushchev/chainervr/ucf101"
video_url = "http://crcv.ucf.edu/data/UCF101/UCF101.rar"
anno_url = "http://crcv.ucf.edu/data/UCF101/UCF101TrainTestSplits-RecognitionTask.zip"
fps = 3.0


def extract_episode(src, dst):
    """Extract images from video. Returns True if successfully extracted, False otherwise"""

    v = None
    try:
        try:
            v = imageio.get_reader(src)
        except imageio.core.fetching.NeedDownloadError:
            imageio.plugins.ffmpeg.download()
            v = imageio.get_reader(src)
        meta = v.get_meta_data()
        src_nframes, src_fps = meta["nframes"], meta["fps"]
        drops = math.floor(1.0 * src_nframes * (fps / src_fps))
        indices = range(0, src_nframes, drops)
        os.makedirs(dst, exist_ok=True)
        i = 0
        with click.progressbar(indices, label="Extracting") as it:
            for index in it:
                for offset in [0, -1, 1]:
                    try:
                        img = v.get_data(index + offset)
                        fn = "frame_%04d.jpg" % i
                        imageio.imwrite(os.path.join(dst, fn), img)
                        break
                    except Exception as e:
                        continue
                i += 1
        return True
    except Exception as e:
        print("Video %s is broken" % src)
        raise e
        return False
    finally:
        if v is not None:
            v.close()


def load_episode(name, num_episodes=None, skip=None):
    import imageio

    if num_episodes is None:
        num_episodes = 5
    if skip is None:
        skip = 1
    category, video = os.path.split(name)
    video_base, video_ext = os.path.splitext(video)
    base_dir = os.path.join(utils.get_dataset_directory("ucf101"), "UCF-101")
    images_dir = os.path.join(base_dir, category, video_base)

    if not os.path.exists(images_dir):
        print("Video '%s' is not yet pre-processed." % name)
        video_path = os.path.join(base_dir, name)
        print(video_path)
        ok = extract_episode(video_path, images_dir)
        assert ok, "failed to extract episode"

    all_image_names = sorted(glob.glob(os.path.join(images_dir, "frame_*.jpg")))
    image_names = []
    for i in range(0, num_episodes * skip, skip):
        image_names.append(all_image_names[i])
    images = []
    for image_name in image_names:
        img = imageio.imread(image_name).astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))  # HWC -> CHW
        images.append(img)
    return np.asarray(images)


def get_ucf101(split, n=None):
    # download annotation
    if n is None:
        n = 1
    elif n not in [1, 2, 3]:
        raise ValueError("n must be 1, 2 or 3")
    if split == "train":
        anno_path = os.path.join("ucfTrainTestlist", "trainlist%02d.txt" % n)
    elif split == "test":
        anno_path = os.path.join("ucfTrainTestlist", "testlist%02d.txt" % n)
    anno = utils.cache_or_load_file(
        "ucf101", anno_url, anno_path,
        lambda p: pd.read_csv(p, sep=" ", names=["filename", "class"]),
    )

    # download videos
    video_root_dir = utils.cache_or_load_file(
        "ucf101", video_url, "UCF-101",
        lambda p: p
    )

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
