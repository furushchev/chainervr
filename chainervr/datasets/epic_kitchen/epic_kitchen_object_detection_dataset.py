#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: lfurushchev <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import csv
import chainer
import numpy as np
import os.path as osp
from chainercv import utils
from epic_kitchen_utils import get_object_detection_images
from epic_kitchen_utils import parse_object_detection_annotation
from epic_kitchen_object_detection_labels import epic_kitchen_object_detection_label_names


class EpicKitchenObjectDetectionDataset(chainer.dataset.DatasetMixin):
    def __init__(self, data_dir="auto", anno_path="auto", split="train",
                 force_download=False, download_timeout=None,
                 skip_no_image=True, skip_no_bbox=False):
        if split not in ["train"]:
            raise ValueError("Split '%s' not available" % split)

        if data_dir == "auto":
            if anno_path == "auto":
                data_dir, anno_path, annotations = get_object_detection_images(
                    split, force_download=force_download, download_timeout=download_timeout)
            else:
                data_dir = get_object_detection_images(
                    split, force_download=force_download, download_timeout=download_timeout)[0]
                annotations = parse_object_detection_annotation(anno_path)
        else:
            if anno_path == "auto":
                raise ValueError("unknown annotation path")
            else:
                annotations = parse_object_detection_annotation(anno_path)

        self.data_dir = data_dir
        self.split = split
        self.annotations = self._filter_annotations(
            annotations, skip_no_image, skip_no_bbox)

    def __len__(self):
        return len(self.annotations)

    def _filter_annotations(self, annotations, skip_no_image=True, skip_no_bbox=False):
        valid_annos = []
        for anno in annotations:
            if skip_no_image:
                pid, vid = anno["participant_id"], anno["video_id"]
                fn = "%010d.jpg" % (anno["frame"])
                img_path = osp.join(self.data_dir, self.split, pid, vid, fn)
                if not osp.exists(img_path):
                    continue
            if skip_no_bbox:
                annos = anno["annotations"]
                if not annos:
                    continue
            valid_annos.append(anno)
        return valid_annos

    def get_example(self, i):
        dic = self.annotations[i]
        img = self._get_image(dic)
        annos = dic["annotations"]
        if len(annos) == 0:
            bbox = np.zeros((0, 4), dtype=np.float32)
            labels = []
        else:
            bbox = []
            labels = []
            for anno in annos:
                for top, left, height, width in anno["bounding_boxes"]:
                    bbox.append([top, left, top+height, left+width])
                    labels.append(anno["noun_class"])
            bbox = np.asarray(bbox, dtype=np.float32)

        labels = np.asarray(labels, dtype=np.int32)

        return tuple([img, bbox, labels])

    def _get_image(self, anno):
        pid, vid = anno["participant_id"], anno["video_id"]
        fn = "%010d.jpg" % (anno["frame"])
        img_path = osp.join(self.data_dir, self.split, pid, vid, fn)
        return utils.read_image(img_path, dtype=np.float32, color=True)


if __name__ == '__main__':
    from chainercv.visualizations import vis_bbox
    import matplotlib.pyplot as plt

    dataset = EpicKitchenObjectDetectionDataset(split="train")

    print "Loaded dataset. length:", len(dataset)
    print "Starting slideshow..."

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plot_init = False
    for i in range(0, len(dataset), 10):
        try:
            img, bbox, label = dataset.get_example(i)
        except Exception as e:
            print i, e
        ax = vis_bbox(img, bbox, label, label_names=epic_kitchen_object_detection_label_names, ax=ax)
        if plot_init is False:
            plt.show()
            plot_init = True
        else:
            fig.canvas.draw()
            fig.canvas.flush_events()
        plt.pause(0.1)
        ax.clear()
