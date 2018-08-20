#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Yuki Furuta <furushchev@jsk.imi.i.u-tokyo.ac.jp>

import filelock
import glob as G
import hashlib
import os
import shutil
from six.moves.urllib import request
import sys
import tempfile
import time

from chainer.dataset import download


def get_dataset_directory(dataset_name):
    return download.get_dataset_directory(
        os.path.join("furushchev", "chainervr", dataset_name))


def _reporthook(count, block_size, total_size):
    global start_time
    global last_duration
    if count == 0:
        start_time = time.time()
        last_duration = 0
        print('  %   Total    Recv       Speed  Time left')
        return
    duration = time.time() - start_time
    if duration - last_duration < 1.0:
        return
    else:
        last_duration = duration
    progress_size = count * block_size
    try:
        speed = progress_size / duration
    except ZeroDivisionError:
        speed = float('inf')
    percent = progress_size / total_size * 100
    eta = int((total_size - progress_size) / speed)
    sys.stdout.write(
        '\r{:3.0f} {:4.0f}MiB {:4.0f}MiB {:6.0f}KiB/s {:4d}:{:02d}:{:02d}'
        .format(
            percent, total_size / (1 << 20), progress_size / (1 << 20),
            speed / (1 << 10), eta // 60 // 60, (eta // 60) % 60, eta % 60))
    sys.stdout.flush()


def cached_download(url, cached_path=None):
    """Downloads a file and caches it.

    This is different from the original
    :func:`~chainer.dataset.cached_download` in that the download
    progress is reported.

    It downloads a file from the URL if there is no corresponding cache. After
    the download, this function stores a cache to the directory under the
    dataset root (see :func:`set_dataset_root`). If there is already a cache
    for the given URL, it just returns the path to the cache without
    downloading the same file.

    Args:
        url (string): URL to download from.

    Returns:
        string: Path to the downloaded file.

    """
    cache_root = os.path.join(download.get_dataset_root(), '_dl_cache')
    try:
        os.makedirs(cache_root)
    except OSError:
        if not os.path.exists(cache_root):
            raise

    urlhash = hashlib.md5(url.encode('utf-8')).hexdigest()
    if cached_path is None:
        cached_path = os.path.join(cache_root, urlhash)
    lock_path = cached_path + ".lock"

    with filelock.FileLock(lock_path):
        if os.path.exists(cached_path):
            return cached_path

    temp_root = tempfile.mkdtemp(dir=cache_root)
    try:
        temp_path = os.path.join(temp_root, 'download.cache')
        print('Downloading ...')
        print('From: {:s}'.format(url))
        print('To: {:s}'.format(cached_path))
        request.urlretrieve(url, temp_path, _reporthook)
        with filelock.FileLock(lock_path):
            shutil.move(temp_path, cached_path)
    finally:
        shutil.rmtree(temp_root)

    return cached_path


def extract_archive(path, dest, ext=None):
    """Unarchive file
    Args:
        path (string): Path to extract from.
        dest (string): Path to extract to.
        ext (string): Extension as a hint.
                      If None, extension is estimated from path.
    Returns:
        string: Path to the root directory of extracted objects.
    """

    path = os.path.abspath(path)
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        print("Created %s" % os.path.dirname(path))

    if ext is None:
        _, ext = os.path.splitext(path)
        ext = ext.lower()
    if ext == ".zip":
        from zipfile import ZipFile
        opener = lambda f: ZipFile(f, "r")
    elif ext == ".rar":
        from rarfile import RarFile
        opener = lambda f: RarFile(f, "r")
    elif ext == ".tar":
        from tarfile import TarFile
        opener = lambda f: TarFile(f, "r")
    elif ext in [".gz", ".tgz", ".tar.gz"]:
        from tarfile import TarFile
        opener = lambda f: TarFile(f, "r:gz")
    else:
        raise RuntimeError("could not detect extension")

    with opener(path) as f:
        namelist = f.namelist()
        print("Extracting %s (%d files)" % (path, len(namelist)))
        f.extractall(dest)
        print("Extracted to %s" % dest)

        return os.path.join(dest, os.path.commonpath(namelist))


def cache_or_load_file(dataset, url, path, load, ext=None, cached_path=None, glob=False):
    """Load file if already cached, otherwise download from remote path.
    Args:
        dataset (string): Dataset name
        url (string): URL to download from if no cache found.
        path (string): Path to the loading file
        load (function): Function to load file that takes file path as the first argument.
        ext (string): Extension used for extracting downloaded files.
                      If None, extension is estimated from url.
        cached_path (string): Path to cache destination of downloaded file.
                              If None, md5sum of url is used.
        glob (bool): If glob is True, path is used as glob expression.
    Returns:
        object: Loaded object
    """

    if ext is None:
        _, ext = os.path.splitext(url)
        ext = ext.lower()

    if load is None:
        load = lambda p: p

    dataset_dir = get_dataset_directory(dataset)
    path = os.path.join(dataset_dir, path)

    if glob and not G.glob(path):
        need_download = True
    else:
        need_download = not os.path.exists(path)

    if need_download:
        print("Local file '%s' not found" % path)
        cached_path = cached_download(url, cached_path=cached_path)
        try:
            extract_archive(cached_path, dataset_dir, ext)
        except RuntimeError:
            dest_path = os.path.join(dataset_dir, os.path.basename(path))
            shutil.copy(cached_path, dest_path)

    if glob:
        return [load(p) for p in G.glob(path)]
    else:
        return load(path)
