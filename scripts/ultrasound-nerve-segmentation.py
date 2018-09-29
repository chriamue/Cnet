#!/usr/bin/env python
import os
import random
import shutil
import sys
import zipfile

def unzip(kaggledatapath, datapath):
    data_folder = datapath
    competition_path = kaggledatapath
    train_archive_file = "{}.zip".format("train")
    train_archive_path = os.path.join(competition_path, train_archive_file)
    test_archive_file = "{}.zip".format("test")
    test_archive_path = os.path.join(competition_path, test_archive_file)

    train_path = os.path.join(data_folder, "train/")
    test_path = os.path.join(data_folder, "test/")

    if not os.path.exists(train_path):
        zip_ref = zipfile.ZipFile(train_archive_path, 'r')
        zip_ref.extractall(data_folder)
        zip_ref.close()
        print('Train Data extracted.')
    if not os.path.exists(test_path):
        zip_ref = zipfile.ZipFile(test_archive_path, 'r')
        zip_ref.extractall(data_folder)
        zip_ref.close()
        print('Test Data extracted.')

# source: https://github.com/milesial/Pytorch-UNet/blob/master/utils/load.py
def get_ids(dir):
    """Returns a list of the ids in the directory"""
    return (f[:-4] for f in os.listdir(dir) if "mask" not in f)

# source: https://github.com/milesial/Pytorch-UNet/blob/master/utils/load.py
def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return dataset[:-n], dataset[-n:]

def split_val(datapath):
    """splits dataset in train and val set and copies files to 'val' and 'val_masks' folders."""
    val_path = os.path.join(datapath, "val/")
    train_path = os.path.join(datapath, "train/")
    val_masks_path = os.path.join(datapath, "val_masks/")
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(val_masks_path):
        os.makedirs(val_masks_path)
    dataset = get_ids(train_path)
    train_ids ,val_ids =split_train_val(dataset, val_percent=0.05)
    for id in val_ids:
        source = os.path.join(train_path, id + ".tif")
        dest = os.path.join(val_path, id + ".tif")
        shutil.move(source, dest)
        source_mask = os.path.join(train_path, id + "_mask.tif")
        dest_mask = os.path.join(val_masks_path, id + "_mask.tif")
        shutil.move(source_mask, dest_mask)

def move_train_masks(datapath):
    """moves mask files to 'train_masks' folder"""
    train_path = os.path.join(datapath, "train/")
    train_masks_path = os.path.join(datapath, "train_masks/")
    if not os.path.exists(train_masks_path):
        os.makedirs(train_masks_path)
    dataset = get_ids(train_path)
    for id in dataset:
        source_mask = os.path.join(train_path, id + "_mask.tif")
        dest_mask = os.path.join(train_masks_path, id + "_mask.tif")
        shutil.move(source_mask, dest_mask)

if __name__ == "__main__":
    datapath = '../data/'
    kaggledatapath = sys.argv[1]
    if len(sys.argv) == 3:
        datapath= sys.argv[2]
    unzip(kaggledatapath, datapath)
    split_val(datapath)
    move_train_masks(datapath)