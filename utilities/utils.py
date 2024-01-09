# system modules
import os
import datetime
import json
from statistics import mean

# installed modules
import torch
import numpy as np
import cv2 as cv
from tqdm import tqdm
import pathlib
from torch.nn.utils import clip_grad
from scipy import ndimage

parent_directory = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent


def get_data_from_txt_file(text_file_path: str = None):
    """
    Get all IDs from the text file.
    Args.:  <str>text_file_path: path to the text file to be read from.
    Return: <list>ids: list of IDs returned from the text file
    """
    if text_file_path is None:
        raise ValueError(
            "`text_file_path` parameter is invalid. Provided path is: ", text_file_path)

    if os.path.exists(text_file_path) is False:
        raise ValueError(
            "`text_file_path` does not exist in storage. Provided path is: ", text_file_path)

    ids = list()
    with open(text_file_path) as text_file:
        ids = [line.rstrip() for line in text_file]

    if len(ids) > 0:
        return ids

    else:
        raise ValueError("`text_file_path` contains no ids")


def get_part_instance_masks_and_BBs_with_classes(segmentation: torch.Tensor = None,
                                            smallest_bb_area: int = 10,
                                            helper_data: dict = None,
                                            segmentation_path: str = None):
    """
    Get/extract segmentation masks and bounding boxes from segmentations
    Ref.: https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

    Args.:  <torch.Tensor>segmentation: segmentation of shape [1,H,W]
            <int>smallest_bb_area: the smallest allowable bounding box area
            <str>segmentation_path: name of the segmentation image for debugging purposes

    Return: <dict> `instance_ids`, `instance_masks`, and `bounding_boxes_n_classes`
    """
    # security checkpoint - you shall not pass hahahahaha
    if isinstance(segmentation, torch.Tensor) is False:
        raise TypeError("Invalid value for `segmentations` parameter")

    if segmentation_path is None:
        raise ValueError("Please specify your segmentaionf file path")

    obj_ids = torch.unique(segmentation)  # get unique segmentations
    # obj_ids_plus_bkgrnd = obj_ids.copy()
    obj_ids = obj_ids[1:]  # remove the background class

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = segmentation.type(torch.uint8) == obj_ids[:, None, None]
    masks = masks.type(torch.uint8)

    # genereate valid bounding boxes only if we have masks
    assert(masks.shape[0] > 0), "No masks available for {}".format(
        segmentation_path)
    mask_2_bb = masks_to_boxes(masks)  # get bounding boxes
    assert(masks.shape[0] == mask_2_bb.shape[0]
           ), "Masks count do not correspond to BB count for {}".format(segmentation_path)

    # Get only the valid bounding boxes. The boxes are of shape (x1, y1, x2, y2), where (x1, y1)
    # specify the top-left box corner, and (x2, y2) specify the bottom-right box corner.
    # see https://pytorch.org/vision/main/generated/torchvision.ops.masks_to_boxes.html and
    # https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/boxes.py for details
    hw = segmentation.shape[1:]  # (H,W) dimension of the image or segmentation
    box_list, mask_list, class_list, human_id_list = [], [], [], []
    for idx in range(len(mask_2_bb)):
        bbox = mask_2_bb[idx]
        mask = masks[idx]

        inst_class_id = obj_ids[idx]
        part_n_human_id = helper_data[int(inst_class_id)]
        # convert list to np for easy convertion to torch.Tensor
        part_n_human_id = np.array(part_n_human_id)
        part_n_human_id = torch.from_numpy(part_n_human_id)
        part_id, human_id = part_n_human_id

        #  part_id -= 1 # take care of indexing
        #  human_id -= 1 # take care of indexing

        x1, y1, x2, y2 = bbox

        if (0 <= x1 < x2) and (0 <= y1 < y2) is False:
            continue

        box_list.append(bbox)
        mask_list.append(mask)
        class_list.append(part_id)
        human_id_list.append(human_id)

    # type cast for torch operations
    box_list = torch.stack(box_list)
    mask_list = torch.stack(mask_list)
    class_list = torch.stack(class_list)
    human_id_list = torch.stack(human_id_list)

    box_list = clip_boxes(box_list, hw)  # clip boxes to remove negatives
    box_list, mask_list, class_list, human_id_list = remove_small_box(box_list, mask_list, class_list, human_id_list, smallest_bb_area)

    # create a canvas for bounding boxes.
    # The shape of this canvas is (num_boxes,6),
    # where 5 represents (x1, y1, x2, y2,object_or_class_ID,human_id)
    bounding_boxes = torch.zeros((len(box_list), 6))
    bounding_boxes[:, :4] = box_list
    bounding_boxes[:, -2] = class_list
    bounding_boxes[:, -1] = human_id_list

    # # re-assemble the masks, taking into account all the smaller masks
    # # that have been removed.
    # instance_masks = merge_masks_with_instances(mask_list,class_list) # dim = (H,W)
    # instance_masks = instance_masks[None,:,:] # expand dimension to become (C,H,W)

    return {"instance_masks": mask_list, "bounding_boxes_n_classes": bounding_boxes}


def get_human_instance_masks_and_BBs_with_classes(segmentation: torch.Tensor = None,
                                            smallest_bb_area: int = 10,
                                            list_of_classes: list = [],
                                            segmentation_path: str = None):
    """
    Get/extract segmentation masks and bounding boxes from segmentations
    Ref.: https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html

    Args.:  <torch.Tensor>segmentation: segmentation of shape [1,H,W]
            <int>smallest_bb_area: the smallest allowable bounding box area
            <str>segmentation_path: name of the segmentation image for debugging purposes

    Return: <dict> `instance_ids`, `instance_masks`, and `bounding_boxes_n_classes`
    """
    # security checkpoint - you shall not pass hahahahaha
    if isinstance(segmentation, torch.Tensor) is False:
        raise TypeError("Invalid value for `segmentations` parameter")

    if segmentation_path is None:
        raise ValueError("Please specify your segmentaionf file path")

    if len(list_of_classes) == 0:
        raise ValueError("Please provide the class list")

    obj_ids = torch.unique(segmentation)  # get unique segmentations
    # obj_ids_plus_bkgrnd = obj_ids.copy()
    obj_ids = obj_ids[1:]  # remove the background class

    # split the color-encoded mask into a set of boolean masks.
    # Note that this snippet would work as well if the masks were float values instead of ints.
    masks = segmentation.type(torch.uint8) == obj_ids[:, None, None]

    # genereate valid bounding boxes only if we have masks
    assert(masks.shape[0] > 0), "No masks available for {}".format(segmentation_path)
    mask_2_bb = masks_to_boxes(masks)  # get bounding boxes
    assert(masks.shape[0] == mask_2_bb.shape[0]), "Masks count do not correspond to BB count for {}".format(segmentation_path)

    # Get only the valid bounding boxes. The boxes are of shape (x1, y1, x2, y2), where (x1, y1)
    # specify the top-left box corner, and (x2, y2) specify the bottom-right box corner.
    # see https://pytorch.org/vision/main/generated/torchvision.ops.masks_to_boxes.html and
    # https://github.com/facebookresearch/Detectron/blob/main/detectron/utils/boxes.py for details
    hw = segmentation.shape[1:]  # (H,W) dimension of the image or segmentation
    box_list, mask_list, class_list, human_id_list = [], [], [], []
    for idx in range(len(mask_2_bb)):
        bbox = mask_2_bb[idx]
        mask = masks[idx]

        x1, y1, x2, y2 = bbox

        if (0 <= x1 < x2) and (0 <= y1 < y2) is False:
            continue

        box_list.append(bbox)
        mask_list.append(mask)
        # there is just 1 class for humans
        class_list.append(torch.from_numpy(np.array(1, dtype=np.int64)))
        human_id_list.append(obj_ids[idx])

    # type cast for torch operations
    box_list = torch.stack(box_list)
    mask_list = torch.stack(mask_list)
    class_list = torch.stack(class_list)
    human_id_list = torch.stack(human_id_list)

    box_list = clip_boxes(box_list, hw)  # clip boxes to remove negatives
    box_list, mask_list, class_list, human_id_list = remove_small_box(
        box_list, mask_list, class_list, human_id_list, smallest_bb_area)

    # create a canvas for bounding boxes.
    # The shape of this canvas is (num_boxes,6),
    # where 5 represents (x1, y1, x2, y2,object_or_class_ID,human_id)
    bounding_boxes = torch.zeros((len(box_list), 6))
    bounding_boxes[:, :4] = box_list
    bounding_boxes[:, -2] = class_list
    bounding_boxes[:, -1] = human_id_list

    # # re-assemble the masks, taking into account all the smaller masks
    # # that have been removed.
    # instance_masks = merge_masks_with_instances(mask_list,class_list) # dim = (H,W)
    # instance_masks = instance_masks[None,:,:] # expand dimension to become (C,H,W)

    return {"instance_masks": mask_list, "bounding_boxes_n_classes": bounding_boxes}


def merge_masks_with_instances(masks: list = None, instance_ids: list = None):
    """
    Code and example posted to https://stackoverflow.com/a/76531685/3901871 for safe keeping
    """
    if masks.shape[0] != instance_ids.shape[0]:
        raise ValueError("Dimentionalitty conflict")

    # just make sure all data types are int
    masks = [mask.type(torch.uint8) for mask in masks]
    # we use .uint8 because we are sure classes are limited to 59 at most for MHP dataset
    instance_ids = [iID.type(torch.uint8) for iID in instance_ids]

    # Create an empty tensor to store the merged masks
    merged_mask = torch.zeros_like(masks[0]).type(torch.uint8)

    # Iterate over each mask and its corresponding instance ID
    for instance_id, mask in zip(instance_ids, masks):
        # Apply the instance mask to the current mask
        instance_mask = torch.where(mask > 0, instance_id, torch.tensor(0))
        merged_mask = torch.max(merged_mask, instance_mask)

    return merged_mask


def resize_image(img: np.array, size: tuple = (300, 300), interpolation: str = 'nearest'):
    """
    Resize image and place on a black canvas to get a square shape.
    Code adopted from https://stackoverflow.com/a/49208362/3901871

    In resizing labels, the type of iterpolation is important. A wrong interpolation
    will result in loosing the instance segmentations. See discussion on this at
    https://stackoverflow.com/a/67076228/3901871. For RGB images, use any desired
    interpolation method, but for labels, use nearest neighbor. In the resize_image()
    method, `nearest` is set as the default interpolation method.
    """

    interpolations = {'nearest':cv.INTER_NEAREST,
                        'bilinear': cv.INTER_LINEAR,
                        'bicubic': cv.INTER_CUBIC}

    h, w, c = img.shape

    if h == w:
        return cv.resize(img, size, interpolation=interpolations[interpolation])

    dif = h if h > w else w
    x_pos = (dif - w)//2
    y_pos = (dif - h)//2

    canvas = np.zeros((dif, dif, c), dtype=img.dtype)
    canvas[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]

    return cv.resize(canvas, size, interpolation=interpolations[interpolation])


def clip_boxes(boxes, hw):
        """
        Clip (limit) the values in a bounding box array. Given an interval, values outside the
        interval are clipped to the interval edges. For example, if an interval of [0, 1] is
        specified, values smaller than 0 become 0, and values larger than 1 become 1.
        """
        boxes[:, [0, 2]] = torch.clamp(boxes[:, [0, 2]], min=0, max=hw[1] - 1)
        boxes[:, [1, 3]] = torch.clamp(boxes[:, [1, 3]], min=0, max=hw[0] - 1)
        return boxes


def remove_small_box(boxes, masks, labels, human_ids, area_limit):
    box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    keep = box_areas > area_limit
    return boxes[keep], masks[keep], labels[keep], human_ids[keep]


def normalize_BB_to_01(bounding_boxes, image_dimension):
        """Normalizes bounding boxes (BB) to intervals between 0 and 1"""
        h, w = image_dimension
        bounding_boxes[:, [0, 2]] = torch.div(bounding_boxes[:, [0, 2]], w)
        bounding_boxes[:, [1, 3]] = torch.div(bounding_boxes[:, [1, 3]], h)
        return bounding_boxes


def get_timestamp_as_str():
    "get date and time as string for model name "
    current_datetime = datetime.datetime.now()
    current_date = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    current_date = current_date.replace("-", "")
    current_date = current_date.replace(" ", "_")
    current_date = current_date.replace(":", "")
    return str(current_date)


def generate_norm_file(list_of_dictionaries: torch.Tensor, save_dir: str = "./"):
    """
    Args.:  <list>list_of_dictionaries: list of dictionary objects that contain image tensors. 
                                        Then key to the image tensors is `images`,
                                        and each tensor must be of shape [3,H,W]
            <str>:save_dir: directory to where json file should be saved
    """
    if len(list_of_dictionaries) < 1:
        raise Exception("No list availabe")
    
    mean_R, mean_G, mean_B = [], [], []
    std_R, std_G, std_B = [], [], []

    for data in tqdm(list_of_dictionaries):
        images = data["images"]
        images = images.squeeze()
        r = images[0,:,:].type(torch.float32)
        g = images[1,:,:].type(torch.float32)
        b = images[2,:,:].type(torch.float32)
        
        # all means
        mean_R.append(torch.mean(r).item())
        mean_G.append(torch.mean(g).item())
        mean_B.append(torch.mean(b).item())
        
        # all stds
        std_R.append(torch.std(r).item())
        std_G.append(torch.std(g).item())
        std_B.append(torch.std(b).item())
    
    data = {"means":[mean(mean_R),mean(mean_G),mean(mean_B)],
            "stds":[mean(std_R),mean(std_G),mean(std_B)]}

    file_name  = os.path.join(save_dir,"norm_file.json")
    with open(file_name,"w+") as fp:
        json.dump(data,fp)
    print("Normalization file saved to ",file_name)



def get_norm_values_from_file(file_path:str="normal_file.json"):
    
    if not os.path.exists(file_path):
        raise ValueError("Invalid file path: " + file_path)
    
    with open(file_path,"r") as fp:
        data = json.load(fp)
    
    return data


def unnormalize_image(image, mean, std):
    image = image.transpose(1,2,0)
    image = (image * std) + mean
    image = image.transpose(2,0,1)
    return image


def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """
    Taken from https://pytorch.org/vision/main/_modules/torchvision/ops/boxes.html#masks_to_boxes
    Compute the bounding boxes around the provided masks.

    Returns a [N, 4] tensor containing bounding boxes. The boxes are in ``(x1, y1, x2, y2)`` format with
    ``0 <= x1 < x2`` and ``0 <= y1 < y2``.

    Args:
        masks (Tensor[N, H, W]): masks to transform where N is the number of masks
        and (H, W) are the spatial dimensions.

    Returns:
        Tensor[N, 4]: bounding boxes
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.float)

    n = masks.shape[0]

    bounding_boxes = torch.zeros((n, 4), device=masks.device, dtype=torch.float)

    for index, mask in enumerate(masks):
        y, x = torch.where(mask != 0)

        bounding_boxes[index, 0] = torch.min(x)
        bounding_boxes[index, 1] = torch.min(y)
        bounding_boxes[index, 2] = torch.max(x)
        bounding_boxes[index, 3] = torch.max(y)

    return bounding_boxes


def clip_grads(params):
    params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
    if len(params) > 0:
        return clip_grad.clip_grad_norm_(params, max_norm=35, norm_type=2)

def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def gradinator(x):
    x.requires_grad = False
    return x

def get_lr(optimizer):
    return optimizer.param_groups[0]['lr']


def get_warmup_lr(cur_iters, warmup_iters, bash_lr, warmup_ratio, warmup='linear'):

    if warmup == 'constant':
        warmup_lr = bash_lr * warmup_ratio 
    elif warmup == 'linear':
        k = (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
        warmup_lr = bash_lr * (1 - k)
    elif warmup == 'exp':
        k = warmup_ratio**(1 - cur_iters / warmup_iters)
        warmup_lr = bash_lr * k
    return warmup_lr


def show_result_ins(img,
                    result,
                    score_thr=0.3,
                    sort_by_density=False,
                    class_list=[],
                    color_pallet=None):
    if isinstance(img, str):
        img = cv.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape
    
    # print(result[0])
    
    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_inds = score > score_thr
    seg_label = seg_label[vis_inds]
    num_mask = seg_label.shape[0]
    cate_label = cate_label[vis_inds]
    cate_score = score[vis_inds]

    if sort_by_density:
        mask_density = []
        for idx in range(num_mask):
            cur_mask = seg_label[idx, :, :]
            cur_mask = imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.int32)
            mask_density.append(cur_mask.sum())
        orders = np.argsort(mask_density)
        seg_label = seg_label[orders]
        cate_label = cate_label[orders]
        cate_score = cate_score[orders]

    np.random.seed(42)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]
    #img_show = None
    for idx in range(num_mask):
        idx = -(idx+1)
        cur_mask = seg_label[idx, :, :]
        cur_mask = imresize(cur_mask, (w, h))
        cur_mask = (cur_mask > 0.5).astype(np.uint8)
        if cur_mask.sum() == 0:
            continue
        color_mask = color_masks[idx]
        cur_mask_bool = cur_mask.astype(np.bool_)
        img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

        cur_cate = cate_label[idx]
        cur_score = cate_score[idx]

        label_text =  class_list[cur_cate] 
        label_text += '|{:.02f}'.format(cur_score)
        center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
        vis_pos = (max(int(center_x) - 10, 0), int(center_y))
        cv.putText(img_show, label_text, vis_pos,
                        cv.FONT_HERSHEY_COMPLEX, 0.3, (255, 255, 255))  # green
 
    return img_show


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None):
    """Resize image to a given size.
    Args:
        img (ndarray): The input image.
        size (tuple[int]): Target size (w, h).
        return_scale (bool): Whether to return `w_scale` and `h_scale`.
        interpolation (str): Interpolation method, accepted values are
            "nearest", "bilinear", "bicubic", "area", "lanczos" for 'cv2'
            backend, "nearest", "bilinear" for 'pillow' backend.
        out (ndarray): The output destination.
    Returns:
        tuple | ndarray: (`resized_img`, `w_scale`, `h_scale`) or
            `resized_img`.
    """
    interpolations = {'nearest':cv.INTER_NEAREST,
                        'bilinear': cv.INTER_LINEAR,
                        'bicubic': cv.INTER_CUBIC}
    
    h, w = img.shape[:2]
    
    resized_img = cv.resize(
            img, size, dst=out, interpolation=interpolations[interpolation])
    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale
    

def clean_data_recieved_from_collator(bbs_n_labels:None,instance_masks:None,computing_device:None):
    
    # some house keeping
    if computing_device is None:
        raise ValueError("Invalid parameter for `computing_device`")

    if bbs_n_labels.shape[0] != instance_masks.shape[0]:
        raise ValueError("Invalid parameters for either `bbs_n_labels` or `instance_masks`")

    fetched_batch = bbs_n_labels.shape[0]
    gt_bboxes = []
    gt_labels = []
    gt_masks = []
    for idx in range(fetched_batch):
        current_bbs_n_labels = bbs_n_labels[idx, :, :]
        current_bbs_n_labels = current_bbs_n_labels[current_bbs_n_labels[:, -1] != -1] # remove all negative vaulues that were used for padding
        
        # prepare gt bounding boxes
        bbox = current_bbs_n_labels[:,:4]
        bbox = gradinator(bbox.to(computing_device))
        gt_bboxes.append(bbox)

        # prepare gt labels
        label = current_bbs_n_labels[:,4].long()
        num_valid_labels = label.shape[0]
        label = gradinator(label.to(computing_device))
        gt_labels.append(label)

        # prepare GT masks
        # lets simply use the number of labels to fetch the number of valid masks
        # remember we have some labels that have -1 values for collation padding purposes
        masks = instance_masks[idx]
        masks = masks[:num_valid_labels,:,:] # fetch only valid masks. All arrays with -1 values are removed
        masks = masks.numpy().astype(np.uint8)
        gt_masks.append(masks)
    
    return gt_bboxes, gt_labels, gt_masks



def get_model_size(model, input_size):
    # # Initialize a dummy input tensor with the specified size
    # # input_size dim = (1, 3, 224, 224) example
    # dummy_input = torch.randn(input_size)
    
    # # Serialize the model to a binary buffer
    # buffer = torch.jit.trace(model, dummy_input).save_to_buffer()
    
    # # Calculate the size in bytes and convert it to megabytes
    # model_size_bytes = len(buffer)
    # model_size_megabytes = model_size_bytes / (1024 ** 2)
    
    # print(f"Model size: {model_size_megabytes:.2f} MB")
    total_size_bytes = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(b.numel() for b in model.buffers())
    print(f"Model size in bytes: {total_size_bytes}")

    # Convert bytes to megabytes
    total_size_megabytes = total_size_bytes / (1024 * 1024)
    print(f"Model size in megabytes: {total_size_megabytes:.2f} MB")
    
