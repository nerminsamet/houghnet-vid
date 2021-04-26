# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging

from . import concat_dataset as D
from src.lib.datasets.dataset_factory import get_dataset
from src.lib.datasets.dataset.vid import VIDDataset
import os

DATA_LIST_TRAIN = ('DET_train_30classes', 'VID_train_15frames')
DATA_LIST_VAL = ('VID_val_videos',)

DATASETS = {
    "DET_train_30classes": {
        "img_dir": "ILSVRC2015/Data/DET",
        "anno_path": "ILSVRC2015/Annotations/DET",
        "img_index": "ILSVRC2015/ImageSets/DET_train_30classes.txt"
    },
    "VID_train_15frames": {
        "img_dir": "ILSVRC2015/Data/VID",
        "anno_path": "ILSVRC2015/Annotations/VID",
        "img_index": "ILSVRC2015/ImageSets/VID_train_15frames.txt"
    },
    "VID_train_every10frames": {
        "img_dir": "ILSVRC2015/Data/VID",
        "anno_path": "ILSVRC2015/Annotations/VID",
        "img_index": "ILSVRC2015/ImageSets/VID_train_every10frames.txt"
    },
    "VID_val_frames": {
        "img_dir": "ILSVRC2015/Data/VID",
        "anno_path": "ILSVRC2015/Annotations/VID",
        "img_index": "ILSVRC2015/ImageSets/VID_val_frames.txt"
    },
    "VID_val_videos": {
        "img_dir": "ILSVRC2015/Data/VID",
        "anno_path": "ILSVRC2015/Annotations/VID",
        "img_index": "ILSVRC2015/ImageSets/VID_val_videos.txt"
    }
}

def get_data(name, data_dir, method="base"):
    dataset_dict = {
        "base": "VIDDataset",
        "mega": "VIDMEGADataset",
    }
    if ("DET" in name) or ("VID" in name):
        # data_dir =  DATA_DIR
        attrs =  DATASETS[name]
        args = dict(
            image_set=name,
            data_dir=data_dir,
            img_dir=os.path.join(data_dir, attrs["img_dir"]),
            anno_path=os.path.join(data_dir, attrs["anno_path"]),
            img_index=os.path.join(data_dir, attrs["img_index"])
        )
        return dict(
            factory=dataset_dict[method],
            args=args,
        )
    raise RuntimeError("Dataset not available: {}".format(name))


def build_dataset(dt, opt, is_train=True, method="base"):

    if is_train:
        dataset_list = DATA_LIST_TRAIN
    else:
        dataset_list = DATA_LIST_VAL

    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError(
            "dataset_list should be a list of strings, got {}".format(dataset_list)
        )
    datasets = []
    for dataset_name in dataset_list:
        data = get_data(dataset_name, opt.data_dir, method)
        # factory = getattr(D, data["factory"])
        args = data["args"]
        if "VID" in data["factory"]:
            args["is_train"] = is_train
        # dataset = VIDDataset(**args)
        dataset = dt(**args, opt=opt)
        datasets.append(dataset)

    # for testing, return a list of datasets
    if not is_train:
        return datasets

    # for training, concatenate all datasets into a single one
    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)

    return dataset






