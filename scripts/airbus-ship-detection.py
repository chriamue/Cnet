#!/usr/bin/env python
import os
import random
import shutil
import sys
import zipfile
import numpy as np
import pandas as pd
import cv2


def unzip(kaggledatapath, datapath):
    data_folder = datapath
    competition_path = kaggledatapath
    train_archive_file = "{}.zip".format("train")
    train_archive_path = os.path.join(competition_path, train_archive_file)
    test_archive_file = "{}.zip".format("test")
    test_archive_path = os.path.join(competition_path, test_archive_file)
    train_masks_archive_file = "{}.zip".format("train_ship_segmentations.csv")
    train_masks_archive_path = os.path.join(
        competition_path, train_masks_archive_file)

    train_path = os.path.join(data_folder, "train/")
    test_path = os.path.join(data_folder, "test/")

    if not os.path.exists(train_path):
        zip_ref = zipfile.ZipFile(train_archive_path, 'r')
        zip_ref.extractall(train_path)
        zip_ref.close()
        print('Train Data extracted.')
    if not os.path.exists(os.path.join(competition_path, "train_ship_segmentations.csv")):
        zip_ref = zipfile.ZipFile(train_masks_archive_path, 'r')
        zip_ref.extractall(data_folder)
        zip_ref.close()
        print('Train Masks Data extracted.')
    if not os.path.exists(test_path):
        zip_ref = zipfile.ZipFile(test_archive_path, 'r')
        zip_ref.extractall(test_path)
        zip_ref.close()
        print('Test Data extracted.')

# source: https://www.kaggle.com/kmader/package-segmentation-images


def rle_decode(mask_rle, shape=(768, 768)):
    '''
    mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int)
                       for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T  # Needed to align to RLE direction


def masks_as_image(in_mask_list):
    # Take the individual ship masks and create a single mask array for all ships
    all_masks = np.zeros((768, 768), dtype=np.int16)
    # if isinstance(in_mask_list, list):
    for mask in in_mask_list:
        if isinstance(mask, str):
            all_masks += rle_decode(mask)
    return np.expand_dims(all_masks, -1)


def generate_masks(kaggledatapath, datapath):
    train_masks_archive_file = "{}.csv".format("train_ship_segmentations")
    train_masks_archive_path = os.path.join(
        kaggledatapath, train_masks_archive_file)
    train_masks_path = os.path.join(datapath, "train_masks/")

    if not os.path.exists(train_masks_path):
        os.makedirs(train_masks_path)

        masks = pd.read_csv(train_masks_archive_path)
        l = len(masks['ImageId'])
        for i, id in enumerate(masks['ImageId']):
            img_path = os.path.join(
                train_masks_path, id.split('.')[0] + "_mask.png")
            rle = masks.query('ImageId=="'+id+'"')['EncodedPixels']

            mask = masks_as_image(rle)
            mask[mask > 0] = 255
            cv2.imwrite(img_path, mask)
            print(i, "/", l, id)
        print('masks generated')

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
    train_masks_path = os.path.join(datapath, "train_masks/")
    val_masks_path = os.path.join(datapath, "val_masks/")
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    if not os.path.exists(val_masks_path):
        os.makedirs(val_masks_path)
    dataset = get_ids(train_path)
    train_ids, val_ids = split_train_val(dataset, val_percent=0.05)
    l = len(val_ids)
    for i, id in enumerate(val_ids):
        source = os.path.join(train_path, id + ".jpg")
        dest = os.path.join(val_path, id + ".jpg")
        shutil.move(source, dest)
        source_mask = os.path.join(train_masks_path, id + "_mask.png")
        dest_mask = os.path.join(val_masks_path, id + "_mask.png")
        shutil.move(source_mask, dest_mask)
        print(i, "/", l, id)


def move_train_masks(datapath):
    """moves mask files to 'train_masks' folder"""
    train_path = os.path.join(datapath, "train/")
    train_masks_path = os.path.join(datapath, "train_masks/")
    if not os.path.exists(train_masks_path):
        os.makedirs(train_masks_path)
    dataset = get_ids(train_path)
    for id in dataset:
        source_mask = os.path.join(train_path, id + "_mask.png")
        dest_mask = os.path.join(train_masks_path, id + "_mask.png")
        shutil.move(source_mask, dest_mask)


if __name__ == "__main__":
    datapath = '../data/'
    kaggledatapath = sys.argv[1]
    if len(sys.argv) == 3:
        datapath = sys.argv[2]
    unzip(kaggledatapath, datapath)
    generate_masks(kaggledatapath, datapath)
    split_val(datapath)
    # move_train_masks(datapath)
