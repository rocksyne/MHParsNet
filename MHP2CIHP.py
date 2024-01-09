"""
CREDIT:

Credit to RanTaimu (https://github.com/RanTaimu) for this pre-processing script. Script is 
taken from https://raw.githubusercontent.com/RanTaimu/M-CE2P/master/metrics/MHP2CIHP.py

The annotations in MHP are not grouped into semantic and instance categories. 
Rather for each image, the ground truth annotations are distributed across N images, 
where N denotes the number of human instances in the image. 
This script processes all such annotations into the grouping format of CIHP.
See the README.txt file of the CIHP dataset for more information

For the train and val directories of the MHPV2 dataset, three new directories will be and added.
These directories are:

・ Category_ids: semantic part segmentation labels
・ Human_ids:    semantic person segmentation labels
・ Instance_ids: instance-level human parsing labels

TODO:
    [] Clean up file / code properly
"""

import os
import os.path as osp
import numpy as np
import cv2
import os
from PIL import Image as PILImage
import numpy as np
from tqdm import tqdm
import argparse


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--directory", type=str, default=None, help="MHPV2 parent directory")
args = vars(ap.parse_args())



def gt_process(gt_folder, name_prefix):
    """
    Get Category_ids and Human_ids
    """
    human_num = 1
    while not os.path.exists(os.path.join(gt_folder, '%s_%02d_01.png' % (name_prefix, human_num))):
        human_num += 1

    cat_gt = None
    human_gt = None
    for human_id in range(1, human_num + 1):
        # Label is only put in R channel.
        name = '%s_%02d_%02d.png' % (name_prefix, human_num, human_id)
        single_human_gt = cv2.imread(osp.join(gt_folder, name))[:, :, 2]
        original_shape = single_human_gt.shape
        single_human_gt = single_human_gt.reshape(-1)

        if cat_gt is None:
            cat_gt = np.zeros_like(single_human_gt)
        if human_gt is None:
            human_gt = np.zeros_like(single_human_gt)

        indexes = np.where(single_human_gt != 0)
        cat_gt[indexes] = single_human_gt[indexes]
        human_gt[indexes] = human_id

    assert(cat_gt.max() <= 58)
    assert(human_gt.max() <= 58)

    cat_gt = cat_gt.reshape(original_shape)
    human_gt = human_gt.reshape(original_shape)
    return cat_gt, human_gt


def get_instance(cat_gt, human_gt):
    instance_gt = np.zeros_like(cat_gt, dtype=np.uint8)

    human_ids = np.unique(human_gt)
    bg_id_index = np.where(human_ids == 0)[0]
    human_ids = np.delete(human_ids, bg_id_index)

    total_part_num = 0
    for id in human_ids:
        human_part_label = (np.where(human_gt == id, 1, 0)
                            * cat_gt).astype(np.uint8)
        part_classes = np.unique(human_part_label)

        for part_id in part_classes:
            if part_id == 0:
                continue
            total_part_num += 1
            instance_gt[np.where(human_part_label == part_id)] = total_part_num

    if total_part_num >= 255:
        print("total_part_num exceed: {}".format(total_part_num))
        exit()

    return instance_gt

# credit: https://github.com/RanTaimu/M-CE2P/blob/master/metrics/helper.py
def generate_help_file(split, data_parent_dir):
    """
    Generate help file for groundtruth to accelerate the computation of AP.
    """

    DATA_ROOT = os.path.join(data_parent_dir,split) # 
    GLOBAL_GT_FOLDER = os.path.join(DATA_ROOT, 'Category_ids')
    INSTANCE_GT_FOLDER = os.path.join(DATA_ROOT, 'Instance_ids')
    HUMAN_GT_FOLDER = os.path.join(DATA_ROOT, 'Human_ids')

    IMAGE_NAME_LIST = os.path.join(data_parent_dir,"list", split+".txt")

    with open(IMAGE_NAME_LIST, 'r') as f:
        image_name_list = [x.strip() for x in f.readlines()]

    pbar = tqdm(total=len(image_name_list), desc="Generating help file for "+split+" data split.")
    for count, image_name in enumerate(image_name_list):
        # print('{} / {}: {}'.format(count + 1, len(image_name_list), image_name))
        global_gt_img = PILImage.open(os.path.join(
            GLOBAL_GT_FOLDER, image_name + '.png'))
        human_gt_img = PILImage.open(os.path.join(
            HUMAN_GT_FOLDER, image_name + '.png'))
        instance_gt_img = PILImage.open(os.path.join(
            INSTANCE_GT_FOLDER, image_name + '.png'))
        global_gt_img = np.array(global_gt_img)
        human_gt_img = np.array(human_gt_img)
        instance_gt_img = np.array(instance_gt_img)
        assert(global_gt_img.shape == human_gt_img.shape and
               global_gt_img.shape == instance_gt_img.shape)

        acce_f = open(os.path.join(
            INSTANCE_GT_FOLDER, image_name + '.txt'), 'w')

        instance_ids = np.unique(instance_gt_img)
        background_id_index = np.where(instance_ids == 0)[0]
        instance_ids = np.delete(instance_ids, background_id_index)

        for inst_id in instance_ids:
            inst_id_index = np.where(instance_gt_img == inst_id)
            human_ids = human_gt_img[inst_id_index]
            human_ids = np.unique(human_ids)
            assert(human_ids.shape[0] == 1)

            part_class_ids = global_gt_img[inst_id_index]
            part_class_ids = np.unique(part_class_ids)
            assert(part_class_ids.shape[0] == 1)

            acce_f.write('{} {} {}\n'.format(int(inst_id),
                                             int(part_class_ids[0]),
                                             int(human_ids[0])))
        acce_f.close()
        pbar.update(1)
    pbar.close()


def do_work(SPLIT_TYPE,DATA_ROOT,
            IMAGE_NAME_LIST,
            SRC_CAT_GT_DIR,
            DST_CAT_GT_DIR,
            DST_HUMAN_GT_DIR,
            DST_INST_GT_DIR):

    with open(IMAGE_NAME_LIST, 'r') as f:
        image_name_list = [x.strip() for x in f.readlines()]

    for image_name in tqdm(image_name_list, desc="Processing "+SPLIT_TYPE+" data split..."):
        cat_gt, human_gt = gt_process(SRC_CAT_GT_DIR, image_name)
        instance_gt = get_instance(cat_gt, human_gt)

        cv2.imwrite(osp.join(DST_CAT_GT_DIR, image_name + '.png'),cat_gt)
        cv2.imwrite(osp.join(DST_HUMAN_GT_DIR, image_name + '.png'),human_gt)
        cv2.imwrite(osp.join(DST_INST_GT_DIR, image_name + '.png'),instance_gt)



if __name__ == '__main__':

    if args["directory"] is None or os.path.isdir(args["directory"]) is False:
        raise ValueError("Invalid value for dataset parent diretory")

    print("")
    print("Go make yourself some tea or coffee. This could take a while...more than 30 minutes...")
    print("")
    
    # generate segmentation labels for both train and val datasets
    for split in ['train','val']:
        DATA_ROOT = os.path.join(args["directory"],split)
        SRC_CAT_GT_DIR = osp.join(DATA_ROOT, "parsing_annos")
        IMAGE_NAME_LIST = osp.join(args["directory"],"list", split+".txt")

        DST_CAT_GT_DIR = osp.join(DATA_ROOT, 'Category_ids')
        DST_INST_GT_DIR = osp.join(DATA_ROOT, 'Instance_ids')
        DST_HUMAN_GT_DIR = osp.join(DATA_ROOT, 'Human_ids')

        if not osp.exists(DST_HUMAN_GT_DIR):
            os.makedirs(DST_HUMAN_GT_DIR)
        if not osp.exists(DST_CAT_GT_DIR):
            os.makedirs(DST_CAT_GT_DIR)
        if not osp.exists(DST_INST_GT_DIR):
            os.makedirs(DST_INST_GT_DIR)
        
        # generate the new segmentation data
        do_work(split,DATA_ROOT,IMAGE_NAME_LIST,SRC_CAT_GT_DIR,DST_CAT_GT_DIR,DST_HUMAN_GT_DIR,DST_INST_GT_DIR)
        print("")
        
        # Generate help file for groundtruth to accelerate the computation of APr
        # See https://github.com/RanTaimu/M-CE2P/blob/master/metrics/README.org
        generate_help_file(split,args["directory"])
        print("")
    
    print("")
    print("Pre-processing completed successfully")
    print("")






