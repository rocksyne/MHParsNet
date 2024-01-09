"""
This script performs semantic and instance segmentation predictions on images.
The predictions are saved as colored images. Additionally, it savaes an acceleration 
file by writing sematic classs and corresponding confidence to a text file. This file is
is important when computing the APr, APP and PCP metrics.
"""

import os
import torch
import cv2 as cv
from utilities.utils import get_data_from_txt_file
import numpy as np
from utilities.utils import resize_image
from utilities.helper_utilities import GenerateImageData, ModelStatistics
from tqdm import tqdm
from PIL import Image as PILImage
import time


part_classes = ["Background","Cap/hat", "Helmet", "Face", "Hair", "Left-arm", "Right-arm", "Left-hand", "Right-hand",
                        "Protector", "Bikini/bra", "Jacket/windbreaker/hoodie", "Tee-shirt", "Polo-shirt", 
                        "Sweater", "Singlet", "Torso-skin", "Pants", "Shorts/swim-shorts",
                        "Skirt", "Stockings", "Socks", "Left-boot", "Right-boot", "Left-shoe", "Right-shoe",
                        "Left-highheel", "Right-highheel", "Left-sandal", "Right-sandal", "Left-leg", "Right-leg", "Left-foot", "Right-foot", "Coat",
                        "Dress", "Robe", "Jumpsuit", "Other-full-body-clothes", "Headwear", "Backpack", "Ball", "Bats", "Belt", "Bottle", "Carrybag",
                        "Cases", "Sunglasses", "Eyewear", "Glove", "Scarf", "Umbrella", "Wallet/purse", "Watch", "Wristband", "Tie",
                        "Other-accessary", "Other-upper-body-clothes", "Other-lower-body-clothes"]


data_root = '/media/rockson/Data_drive/datasets/LV-MHP-v2/'
image_dim = (512,512)
saved_model = "/media/rockson/Data_drive/Research/MHP/data/outputs/models/Entire_Model_MHP_20230709_102724_bkup.pt"
computing_device = "cuda"
output_path = os.path.join("data","outputs")


def get_masks_labels_score(seg_result,score_thr=0.3):
    cur_result = seg_result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    return seg_label, cate_label, cate_score


def merge_masks(masks, classes):
    # Initialize the merged mask with zeros
    merged_mask = np.zeros_like(masks[0])

    # Iterate through each mask and merge them
    for mask, class_label in zip(masks, classes):
        # Find pixels belonging to the current class in the mask
        class_pixels = (mask == 1)  # Assuming class index 1 for the current mask

        # Set the corresponding pixels in the merged mask to the class label
        merged_mask[class_pixels] = class_label

    return merged_mask


def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    iou = intersection / union if union > 0 else 0.0
    return iou


def gt_process(pred_human_with_category):
    """
    Get Category_ids and Human_ids
    """
    human_num = pred_human_with_category.shape[0]

    cat_gt = None
    human_gt = None

    for human_id in range(1, human_num + 1):
        # Label is only put in R channel.
        single_human_gt = pred_human_with_category[human_id-1]
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


def compute_confidence(instance_mask, semantic_mask, mapping_dict):
    instance_confidence = {}

    unique_instances = np.unique(instance_mask)
    for instance_id in unique_instances:
        if instance_id in mapping_dict:
            class_id = mapping_dict[instance_id]
            class_mask = semantic_mask == class_id
            instance_pixels = np.sum(instance_mask == instance_id)
            class_pixels = np.sum(class_mask)
            confidence = instance_pixels / class_pixels
            instance_confidence[instance_id] = confidence

    return instance_confidence



def spread_masks(segmentation_mask, num_classes):
    """
    Spreads a segmentation mask across multiple masks, one for each class category.

    Args:
        segmentation_mask (numpy.ndarray): Segmentation mask with class labels.
        num_classes (int): Number of class categories.

    Returns:
        numpy.ndarray: Array of masks, each representing a class category.

    """
    # Create an empty array of masks
    masks = np.zeros((num_classes, *segmentation_mask.shape), dtype=np.uint8)

    # Iterate over each class category
    for class_id in range(0, num_classes):
        # Set the mask where the segmentation mask matches the class label
        masks[class_id] = (segmentation_mask == class_id) * class_id

    return masks



def get_instance(cat_gt, human_gt):
  """
  """
  instance_gt = np.zeros_like(cat_gt, dtype=np.uint8)

  human_ids = np.unique(human_gt)
  bg_id_index = np.where(human_ids == 0)[0]
  human_ids = np.delete(human_ids, bg_id_index)

  class_map = {}

  total_part_num = 0
  for id in human_ids:
      human_part_label = (np.where(human_gt == id, 1, 0)* cat_gt).astype(np.uint8)
      part_classes = np.unique(human_part_label)

      for part_id in part_classes:
          if part_id == 0:
              continue
          total_part_num += 1

          class_map[total_part_num] = part_id
          instance_gt[np.where(human_part_label == part_id)] = total_part_num

  if total_part_num >= 255:
      print("total_part_num exceed: {}".format(total_part_num))
      exit()

  # Make instance id continous.
  ori_cur_labels = np.unique(instance_gt)
  total_num_label = len(ori_cur_labels)
  if instance_gt.max() + 1 != total_num_label:
      for label in range(1, total_num_label):
          instance_gt[instance_gt == ori_cur_labels[label]] = label

  final_class_map = {}
  for label in range(1, total_num_label):
      if label >= 1:
          final_class_map[label] = class_map[ori_cur_labels[label]]

  return instance_gt, final_class_map


def get_palette(num_cls):
  """ 
  Returns the color map for visualizing the segmentation mask.
  Args.: num_cls: Number of classes.
  Returns: The color map.
  """
  n = num_cls
  palette = [0] * (n * 3)
  for j in range(0, n):
    lab = j
    palette[j * 3 + 0] = 0
    palette[j * 3 + 1] = 0
    palette[j * 3 + 2] = 0
    i = 0
    while lab:
      palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
      palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
      palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
      i += 1
      lab >>= 3
  return palette



val_data_id_list_dir = os.path.join(data_root, "list","val.txt")
val_image_dir = os.path.join(data_root, "val", "images")
val_categories_dir = os.path.join(data_root, "val", "Category_ids")
val_humans_dir = os.path.join(data_root, "val", "Human_ids")
val_data_ids = get_data_from_txt_file(val_data_id_list_dir)
palette = get_palette(256)

all_human_images = [os.path.join(val_humans_dir,id+".png") for id in val_data_ids]
all_category_images = [os.path.join(val_categories_dir,id+".png") for id in val_data_ids]
all_images = [os.path.join(val_image_dir,id+".jpg") for id in val_data_ids]


data = GenerateImageData(dimension=image_dim)
model_stat = ModelStatistics()
model = torch.load(saved_model)
model = model.to(computing_device) # computing device
model.eval()


if len(all_human_images) != len(all_category_images):
    raise ValueError("Values do not match")

number_of_frames = 0
times = []
# extract human instances
for human_image, category_image, image in tqdm(list(zip(all_human_images,all_category_images, all_images))):

    # get the file name
    image_id = human_image.split(os.sep)[-1].replace(".png","")

    # get gt category mask
    original_image = cv.imread(image,cv.IMREAD_COLOR) # (H,W,3) where 3 is GBR
    original_image = cv.cvtColor(original_image, cv.COLOR_BGR2RGB) # convert BGR to RGB
    original_image = resize_image(original_image,size=(512,512),interpolation="bilinear")

    # get gt human mask
    gt_human_image = cv.imread(human_image,cv.IMREAD_GRAYSCALE)
    gt_human_image = np.expand_dims(gt_human_image,axis=-1)
    gt_human_image = resize_image(gt_human_image,size=image_dim ,interpolation="nearest") # [H,W] Color channel is ignored again after resize
    gt_human_image[gt_human_image > 58] = 0 # clean the data and make sure foreign classes are eliminated
    gt_unique_human_instances = np.unique(gt_human_image)[1:] # remove the bbackground class
    gt_human_image_masks = gt_human_image.astype(np.uint8) == gt_unique_human_instances [:, None, None] # extract the human instances

    # get gt category mask
    gt_category_image = cv.imread(category_image,cv.IMREAD_GRAYSCALE)
    gt_category_image = np.expand_dims(gt_category_image,axis=-1)
    gt_category_image = resize_image(gt_category_image,size=image_dim ,interpolation="nearest") # [H,W] Color channel is ignored again after resize
    gt_category_image[gt_category_image > 58] = 0 # clean the data and make sure foreign classes are eliminated

    # get image data
    _, normalized, meta_data = data.get_data(image)

    gt_human_with_category = []
    for gt_human_image_mask in gt_human_image_masks:
        msk = gt_human_image_mask*gt_category_image
        gt_human_with_category.append(msk)
    

    with torch.no_grad():
        parts_seg_result, human_seg_result = model(normalized.to(computing_device),operation_mode='evaluate',img_meta=[meta_data]) #normalized shape is [1,3,H,W]

    pred_part_masks, pred_parts_labels, pred_parts_scores = get_masks_labels_score(parts_seg_result)
    pred_human_masks, pred_human_labels, pred_humans_scores = get_masks_labels_score(human_seg_result)

    # avoid any computation is no human mask are predicted
    if len(pred_human_masks)<1:
        print("No valid human mask detected")
        continue

    # the index for `parts_labels` starts from 0, which infact is supposed to be the index of the corresponding label
    # since 0 is the index of the background class originally. In the prediction of `parts_labels`, we skipped the background class.
    # Therefore, although the values of `parts_labels` may start from 0, the 0 is actually the index of the labels (excluding the background class)
    # therefore we will add +1 to each `parts_labels` value to provide the offset. We will skip for `human labels because we have no need for it now`
    pred_parts_labels = [l+1 for l in pred_parts_labels]

    # convert the pred_parts_scores from numpy to lists for 
    # easy manipulation, such as deleting
    pred_parts_scores = pred_parts_scores.tolist()

    # predicted human masks do not have an ordered sequence. That means it is possible to generate masks that are in a totally different order from the ground truth masks
    # An adhoc solution I have in mind is to take each of the predicted human masks and find which of the human instances in the ground truth it overlaps the most with.
    # Here is an example. Assuming we have two ground truth data. Person 1 on the left in mask 1, and Person 2 on the right in mask 2. Now assume we predict human instance masks
    # but Person 1 on the left is in mask 2, and and Person 2 on the right is in mask 1. We have to find a propoer odering mechanism on which mask should come first for
    # dimension compactibility. Therefore, the plan is to find the index of the gt masks which the predicted masks overlap most with, and then arrange the pred masks in that order.
    zero_data = np.zeros(image_dim).astype(np.uint8) # make a blank canvas. No prediction
    re_ordered_masks = [zero_data]*len(gt_human_image_masks) # blank canvas
    re_ordered_scores = [None]*len(gt_human_image_masks)
    bb_n_score_list = []

    for gt_idx, gt_human_image_mask in enumerate(gt_human_image_masks):

        overlaps = dict()
        for pred_idx, pred_human_mask in enumerate(pred_human_masks):
            iou = calculate_iou(pred_human_mask, gt_human_image_mask)
            overlaps[pred_idx] = iou
        sorted_keys = sorted(overlaps, key=overlaps.get, reverse=True)
        best_overlap_index = sorted_keys[0]
        re_ordered_masks[gt_idx] = pred_human_masks[best_overlap_index]

        # generate bounding boxes for this human mask
        ys, xs = np.where(pred_human_masks[best_overlap_index] > 0)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        bb_n_score = (x1, y1, x2, y2,pred_humans_scores[best_overlap_index])
        bb_n_score_list.append(bb_n_score)

    pred_human_masks = np.array(re_ordered_masks) # make np array instead of list. Shape is (H,W,B)

    # now lets merge all the valid predicted parts masks into one mask, and preserve the ids of the instances
    merged_parts_masks = merge_masks(pred_part_masks, pred_parts_labels) 

    # now to get the parts of each human, we simply multiply each human instance with the merged parts instance mask
    pred_human_with_category = list()
    for _, mask in enumerate(pred_human_masks,start=0):
        m = merged_parts_masks*mask
        pred_human_with_category.append(m)
    
    # list of masks human instances and their corresponding body parts
    pred_human_with_category = np.array(pred_human_with_category)

    # Generate global (semantic), human instance and part instance masks
    # Code modified from https://github.com/RanTaimu/M-CE2P/blob/master/metrics/MHP2CIHP.py
    # ======================================================================================
    pred_semantic_mask, pred_humans_mask = gt_process(pred_human_with_category)
    pred_instance_mask, class_map = get_instance(pred_semantic_mask, pred_humans_mask)
    confidence_scores = compute_confidence(pred_instance_mask, pred_semantic_mask, class_map)

    # write sematic classs and corresponding confidence to text file
    with open(os.path.join(output_path,"instance_pred_folder",image_id+".txt"),'w+') as fp:
        for instance_id,confidence_score in confidence_scores.items():
            semantic_class = class_map[instance_id]
            fp.write('{} {}\n'.format(semantic_class, confidence_score*100))
    
    print("Shape of this is: ", pred_semantic_mask.dtype)

    # add color to semantic predictions and save as image
    pred_semantic_mask = PILImage.fromarray(pred_semantic_mask) 
    pred_semantic_mask.putpalette(palette)
    pred_semantic_mask.save(os.path.join(output_path,"semantic_pred_folder",image_id+".png"))

    # save instance mask
    pred_instance_mask = PILImage.fromarray(pred_instance_mask ) 
    pred_instance_mask.putpalette(palette)
    pred_instance_mask.save(os.path.join(output_path,"instance_pred_folder",image_id+".png"))

    # # get gt category mask
    # human_instance_gt = cv.imread("/media/rockson/Data_drive/datasets/LV-MHP-v2/val/Human_ids/"+image_id+".png",cv.IMREAD_GRAYSCALE)
    # human_instance_gt = np.expand_dims(human_instance_gt,axis=-1)
    # human_instance_gt = resize_image(human_instance_gt,size=image_dim ,interpolation="nearest") # [H,W] Color channel is ignored again after resize
    # human_instance_gt[human_instance_gt> 58] = 0 # clean the data and make sure foreign classes are eliminated

    # # save instance mask
    # human_instance_gt = PILImage.fromarray(human_instance_gt ) 
    # human_instance_gt.putpalette(palette)
    # human_instance_gt.save(os.path.join("/home/rockson/Pictures",image_id+"_gt_human.png"))


    # # get gt category mask
    # instance_gt = cv.imread("/media/rockson/Data_drive/datasets/LV-MHP-v2/val/Instance_ids/"+image_id+".png",cv.IMREAD_GRAYSCALE)
    # instance_gt = np.expand_dims(instance_gt,axis=-1)
    # instance_gt = resize_image(instance_gt,size=image_dim ,interpolation="nearest") # [H,W] Color channel is ignored again after resize
    # instance_gt[instance_gt> 58] = 0 # clean the data and make sure foreign classes are eliminated

    # # save instance mask
    # instance_gt = PILImage.fromarray(instance_gt ) 
    # instance_gt.putpalette(palette)
    # instance_gt.save(os.path.join("/home/rockson/Pictures",image_id+"_gt_instance.png"))

    


