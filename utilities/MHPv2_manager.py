"""
Author: Rockson Ayeman (rockson.agyeman@aau.at, rocksyne@gmail.com)
        Bernhard Rinner (bernhard.rinner@aau.at)

For:    Pervasive Computing Group (https://nes.aau.at/?page_id=6065)
        Institute of Networked and Embedded Systems (NES)
        University of Klagenfurt, 9020 Klagenfurt, Austria.

Date:   02.06.2023 (First authored date)

Documentation:
--------------------------------------------------------------------------------
Dataset manager for all Multiple-Human Parsing Dataset v2.0 (MHPv2) 
See https://lv-mhp.github.io/ for details.

TODO: Provide more documentation
"""

# system modules
from __future__ import print_function, division
import os
import pathlib
import glob

# 3rd party modules
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from natsort import natsorted # noqa
from torchvision.transforms import Normalize
import cv2 as cv
import numpy as np


from .utils import get_part_instance_masks_and_BBs_with_classes
from .utils import get_human_instance_masks_and_BBs_with_classes
from .utils import resize_image

# import from custom defined modules
from .utils import get_data_from_txt_file
from .utils import get_norm_values_from_file

project_parent_directory = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent

class MHPv2(Dataset):

    def __init__(self,
                 dimension: tuple = (544, 544),
                 dataset_parent_dir: str = None,
                 dataset_split: str = "train",
                 verify_data_integrity: bool = True,
                 remove_problematic_images: bool = True):
        """
        Prepare the MHPv2 dataset for training and inference. 

        One image in the `images` directory corresponds to multiple annotation files in `parsing_annos` directory.
        The name of each annotation image starts with the same prefix as the image in the `images` directory.
        
        For example, given `train/images/96.jpg`, there are 3 humans in the image. The correspoding annotation files in 
        `train/parsing_annos/` are
            ・ 96_03_01.png, where 96 is the corresponding image name (96.jpg), 3 is # of person, and 1 is person # 1 ID
            ・ 96_03_02.png, where 96 is the corresponding image name (96.jpg), 3 is # of person, and 2 is person # 2 ID
            ・ 96_03_03.png, where 96 is the corresponding image name (96.jpg), 3 is # of person, and 3 is person # 3 ID

        Args.:  <tuple>dimension: output dimension of the dataset
                <str>dataset_split: type of dataset split. Acceptable values [`train`,`validation`]
                <bool>remove_problematic_images: remove images that are known to cause problems

        Return: <dict>samples: batch sample of the dataset. sample includes
                                ・ images, shape is [N,3,H,W]
                                ・ images_path, shape is [N,1]
                                ・ segmentation_label, shape is [N,1,H,W]
                                ・ segmentation_label_path, shape is [N,1]
                                ・ instance_ids, shape is [N]
                                ・ bounding_boxes, shape is [N,5]. 5 is 4 BB co-ordinates and 1 class
                                ・ segmentation_masks, shape is [N,]

                                TODO: get the dimension issue sorted out

        Corrupt images removed:
            1. train/images/1396.jpg
            2. train/images/18613.jpg
            3. train/images/19012.jpg
            4. train/images/19590.jpg
            5. train/images/24328.jpg

        Ref. / Credit: code adopted from https://github.com/RanTaimu/M-CE2P/blob/master/metrics/MHPv2/mhp_data.py

        Images:       images
        Category_ids: semantic part segmentation labels         Categories:   visualized semantic part segmentation labels
        Human_ids:    semantic person segmentation labels       Human:        visualized semantic person segmentation labels
        Instance_ids: instance-level human parsing labels       Instances:    visualized instance-level human parsing labels
        """
        self.dimension = dimension
        self.dataset_split = dataset_split
        self.dataset_parent_dir = dataset_parent_dir
        self.allowed_dataset_split = ["train", "val"]
        self.remove_problematic_images = remove_problematic_images
        self.part_classes = ["Cap/hat", "Helmet", "Face", "Hair", "Left-arm", "Right-arm", "Left-hand", "Right-hand",
                        "Protector", "Bikini/bra", "Jacket/windbreaker/hoodie", "Tee-shirt", "Polo-shirt", 
                        "Sweater", "Singlet", "Torso-skin", "Pants", "Shorts/swim-shorts",
                        "Skirt", "Stockings", "Socks", "Left-boot", "Right-boot", "Left-shoe", "Right-shoe",
                        "Left-highheel", "Right-highheel", "Left-sandal", "Right-sandal", "Left-leg", "Right-leg", "Left-foot", "Right-foot", "Coat",
                        "Dress", "Robe", "Jumpsuit", "Other-full-body-clothes", "Headwear", "Backpack", "Ball", "Bats", "Belt", "Bottle", "Carrybag",
                        "Cases", "Sunglasses", "Eyewear", "Glove", "Scarf", "Umbrella", "Wallet/purse", "Watch", "Wristband", "Tie",
                        "Other-accessary", "Other-upper-body-clothes", "Other-lower-body-clothes"]

        if self.dataset_split not in self.allowed_dataset_split:
            raise ValueError("Invalid value for `dataset_split` parameter")
        
        if dataset_parent_dir is None:
            raise ValueError("Invalid value for `dataset_parent_dir` parameter in")

        # get all essenstial directories
        self.data_id_file = os.path.join(self.dataset_parent_dir, "list", self.dataset_split+".txt")
        self.image_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "images")
        self.parsing_annot_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "parsing_annos")
        self.category_ids_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "Category_ids")
        self.human_ids_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "Human_ids") 
        self.instance_ids_dir = os.path.join(self.dataset_parent_dir, self.dataset_split, "Instance_ids") 
        self.data_ids = get_data_from_txt_file(self.data_id_file)
        
        norm_values = get_norm_values_from_file(file_path=os.path.join(project_parent_directory,"utilities","normal_file.json"))
        self.normalize_image = Normalize(norm_values['means'], norm_values['stds'], inplace=False)
        
        # some debug information
        print("")
        print("[INFO] Dataset name: Multi-human Parsing (MHP) V2")
        print("[INFO] Dataset split: ",self.dataset_split)
        print("[INFO] Number of classes: {:,}. (*please note that this does not include the background class)".format(len(self.part_classes)))
        print("[INFO] Total valid data samples: {:,}".format(len(self.data_ids)))
        print("[INFO] Some image data may have been corrupted during extracting, so remove them if you can")
        print("")

        

    def __len__(self):
        return len(self.data_ids)


    def __getitem__(self, idx):
        image_data_path = os.path.join(self.image_dir,str(self.data_ids[idx])+".jpg")
        image_data = cv.imread(image_data_path,cv.IMREAD_COLOR) # (H,W,3) where 3 is GBR
        image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB) # convert BGR to RGB
        image_data = resize_image(image_data,size=self.dimension,interpolation="bilinear").transpose(2,0,1) # [H,W,3] >> [3,H,W]
        image_data = torch.from_numpy(image_data) # convert to torch array
        image_data = self.normalize_image(image_data.type(torch.float))

        # # # masks and bounding boxes for semantic part segmentation
        # category_seg_annot = read_image(os.path.join(self.category_ids_dir,str(self.data_ids[idx])+".png"),ImageReadMode.GRAY) # (1,H,W)
        # category_seg_annot = resize_image(category_seg_annot,size=self.dimension)
        # # c_ids_data_properties = get_instance_masks_and_BBs_with_classes(category_seg_annot ,image_data_path)
        # # c_masks = c_ids_data_properties['instance_masks']
        # # c_boxes = c_ids_data_properties['bounding_boxes_n_classes']
        # # c_boxes = clip_boxes(c_boxes,self.dimension)
        # # c_boxes = remove_small_box(c_boxes)
        
        
        # masks and bounding boxes for instance-level human parsing
        instance_seg_annot_path = os.path.join(self.instance_ids_dir,str(self.data_ids[idx])+".png")
        instance_seg_annot = cv.imread(instance_seg_annot_path,cv.IMREAD_GRAYSCALE) # (H,W). Color dim for greyscale is always ignored by OpenCV
        instance_seg_annot = np.expand_dims(instance_seg_annot,axis=-1) # (H,W,1). Force add the color channel
        instance_seg_annot = resize_image(instance_seg_annot,size=self.dimension,interpolation="nearest") # [H,W] Color channel is ignored again after resize
        instance_seg_annot = np.expand_dims(instance_seg_annot,axis=0) # (1,H,W). Force add the color channel but keep dim as (C,H,W), so no transpose needed
        instance_seg_annot = torch.from_numpy(instance_seg_annot) # convert to torch tensor

        # get masks and bbs
        helper_data = self._get_helper_data(os.path.join(self.instance_ids_dir,str(self.data_ids[idx])+".txt"))
        i_ids_data_properties = get_part_instance_masks_and_BBs_with_classes(instance_seg_annot,20,helper_data,image_data_path)
        part_instance_masks = i_ids_data_properties['instance_masks']
        part_instance_bbs_n_classes  = i_ids_data_properties['bounding_boxes_n_classes']

        # masks and bounding boxes for semantic person segmentation
        human_seg_annot_path = os.path.join(self.human_ids_dir,str(self.data_ids[idx])+".png")
        human_seg_annot = cv.imread(human_seg_annot_path,cv.IMREAD_GRAYSCALE) # (H,W). Color dim for greyscale is always ignored by OpenCV
        human_seg_annot = np.expand_dims(human_seg_annot,axis=-1) # (H,W,1). Force add the color channel
        human_seg_annot = resize_image(human_seg_annot,size=self.dimension,interpolation="nearest") # [H,W] Color channel is ignored again after resize
        human_seg_annot = np.expand_dims(human_seg_annot,axis=0) # (1,H,W). Force add the color channel but keep dim as (C,H,W), so no transpose needed
        human_seg_annot = torch.from_numpy(human_seg_annot) # convert to torch array
        
        # get masks and BBs
        h_ids_data_properties = get_human_instance_masks_and_BBs_with_classes(human_seg_annot, 20,self.part_classes, image_data_path)
        human_instance_masks = h_ids_data_properties['instance_masks']
        human_bbs_n_classes = h_ids_data_properties['bounding_boxes_n_classes']

        sample = {"images":image_data,
                  "images_path": image_data_path,
                  "part_instance_masks": part_instance_masks,
                  "part_instance_bbs_n_classes": part_instance_bbs_n_classes,
                  "human_instance_masks": human_instance_masks,
                  "human_instance_bbs_n_classes": human_bbs_n_classes
                  }
        return sample


    def validate_images_in_dir(self,image_dir:str = None, file_ext:str = None):
        if image_dir is None:
            raise ValueError("`dir` parameter is invalid")
        
        if file_ext is None or file_ext not in ["jpg", "jpeg", "png"]:
            raise ValueError("`file_ext` parameter is invalid")
        
        assert(os.path.isdir(image_dir)), 'Path does not exist: {}'.format(image_dir)

        # search files of this ext only
        # see https://stackoverflow.com/a/3964689/3901871
        path = os.path.join(image_dir,"*."+file_ext) 
        files = glob.glob(path)

        if len(files) > 0:
            files = natsorted(files)
            files = [os.path.join(image_dir,f) for f in files]

            for file_path in tqdm(files):
                if self.image_is_corrupted(file_path):
                    print(file_path)
        else:
            raise ValueError("Sorry. No `.{}` files found in {}".format(file_ext, image_dir))


    def _get_helper_data(self, file_path:str = None):
        """ 
        The helper text file, named image_ID_number.txt (eg. 3.txt) contains properties of a segmentation file.
        The text file contains 3 colums:
            column 1: All the unique instance ids of a segmentation file (.png) from the Instance_ids directory
            column 2: The human part ID (out of the 58) the unique instance id in column 1 belongs to
            column 3: The ID of the person the ID in column 2 belongs to

        The idea here is to generate a dictionary (key is column 1) that will hold a two element list
        of [human_part_id,human_id], eg. {1:[human_part_id,human_id],...,10:[human_part_id,human_id]},
        where human_part_id is column 1, and human_id is column 2.

        Args.:  <str>file_path: path of the helper file (.txt)
        Return: <dict> eg. {<int>column_1:[<int>column_1,<int>column_2],...}
        """
        output = {}
        all_data = get_data_from_txt_file(file_path)
        
        for data in all_data:
            key, part_id, human_id = data.split() # eg. 2 4 1
            output[int(key)] = [int(part_id), int(human_id)]

        return output



class MHPv2DataCollator(object):
    def __init__(self,image_dimension:tuple=(300,300)):
        """ 
        Read article at the link below to understand the need for this DataCollator class
        https://plainenglish.io/blog/understanding-collate-fn-in-pytorch-f9d1742647d3

        This method will collate data for:
        1. images
        2. labels
        3. instance_ids
        4. instance_masks
        5. bounding_boxes

        The overall objective is, a bacth of data for items 1, 2, 3, 4 and 5
        should all have the same dimensions.

        Again, read the article at the provided URL to fully understand the essense of this.
        https://plainenglish.io/blog/understanding-collate-fn-in-pytorch-f9d1742647d3.

        See link below for extra reading:
        [1] https://androidkt.com/create-dataloader-with-collate_fn-for-variable-length-input-in-pytorch/
        [2] https://www.youtube.com/watch?v=BpuwwyjJHFw

        Code inspired by https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/dataloader.py

        Args.:	<tuple> image_dimension: a tuple of the expected height and width of the output data 

        TODO: Properly document this work
        """
        super(MHPv2DataCollator, self).__init__()
        self.image_dimension = image_dimension

    def __call__(self,data:list=None):

        if not isinstance(data,list):
            raise TypeError("`data` must be of type `list`. the provided type is {}".format(type(data)))
        
        
        # Images
        # =========================================
        images = [d['images'] for d in data]
        if not len(images) > 0:
            raise ValueError("No image present! Please check `images` arguement.")
        
        if (len(images[0].shape) != 3) or (images[0].shape[0] != 3):
            raise ValueError("Dimenstionality incompactibility for `images` parameter. \
                             Shape of image is ".format(images[0].shape))
        
        batch_size = len(images)
        m_height, m_width = self.image_dimension # eg. (512,512). use this to create canvases for the masks
        images = torch.stack(images) # stack all the images to form a batch.

        # Images path
        # =========================================
        images_path = [d['images_path'] for d in data]
        if not len(images_path) > 0:
            raise ValueError("No image data path present! Please check `images_path` arguement.")
        
        if len(images_path) != len(images):
            raise ValueError("The number of images and the path names do not match")
        
        extra_debug_info = self.file_loc_debug_info(images_path)


        # Part instance segmentation masks
        # =========================================
        part_instance_masks = [d['part_instance_masks'] for d in data]
        if not len(part_instance_masks) > 0:
            raise ValueError("No part segmentation mask present! Please check `part_instance_masks` arguement. {}".format(extra_debug_info))
        
        if (len(part_instance_masks[0].shape)!=3): 
            raise ValueError("Mask dimension is not compactible to image dimension, or according to the required shape. \
                             Expected shape is (1,{},{}) but got {}. {}".format(m_height, m_width, part_instance_masks[0].shape,extra_debug_info))
        
        max_num_part_instance_masks = max(len(d) for d in part_instance_masks)
        if max_num_part_instance_masks > 0:
            part_instance_masks_padded = torch.ones((batch_size,max_num_part_instance_masks,m_height, m_width))*-1 # canvas of -1s

            for idx, im in enumerate(part_instance_masks):
                if im.shape[0]>0:
                    part_instance_masks_padded[idx,:im.shape[0],:,:] = im
                
                else:
                    # TODO: clean up this code properly.
                    raise ValueError("Current part instance mask is invalid. {}".format(extra_debug_info))

        else:
            part_instance_masks_padded = torch.ones((batch_size,1,m_height, m_width))*-1 # canvas of -1s
            raise ValueError("Empty part instance mask. {}".format(extra_debug_info))
        

        # Part instance bounding boxes and classes
        # =========================================
        part_instance_bbs_n_classes = [d['part_instance_bbs_n_classes'] for d in data]
        if not len(part_instance_bbs_n_classes) > 0:
            raise ValueError("No part segmentation annotation present! Please check `part_instance_bbs_n_classes` arguement. {}".format(extra_debug_info))
        
        if len(part_instance_bbs_n_classes) != len(part_instance_masks):
            raise ValueError("The number of part instance annotations ({}) do not match the number \
                             of part instance masks ({}). {}".format(len(part_instance_bbs_n_classes), len(part_instance_masks), extra_debug_info))
        
        max_num_part_instance_bbs_n_classes= max(len(d) for d in part_instance_bbs_n_classes)
        if max_num_part_instance_bbs_n_classes > 0:
            part_instance_bbs_n_classes_padded = torch.ones((batch_size,max_num_part_instance_bbs_n_classes,6))*-1 # canvas of -1s

            if max_num_part_instance_bbs_n_classes > 0:
                for idx, bb in enumerate(part_instance_bbs_n_classes):
                    if bb.shape[0]>0:
                        part_instance_bbs_n_classes_padded[idx,:bb.shape[0],:] = bb
                    
                    else:
                        # TODO: clean up this code properly.
                        raise ValueError("Current part instance bounding box is invalid. {}".format(extra_debug_info))
        else:
            part_instance_bbs_n_classes_padded = torch.ones((batch_size,1,6))*-1 # dummy data of -1s
            raise ValueError("There are no bounding boxes for the part instance segmentation masks. {}".format(extra_debug_info))
        

        # Human instannce masks
        # =========================================
        human_instance_masks = [d['human_instance_masks'] for d in data]
        if not len(human_instance_masks) > 0:
            raise ValueError("No human segmentation mask present! Please check `human_instance_masks` arguement. {}".format(extra_debug_info))
        
        if (len(human_instance_masks[0].shape)!=3): 
            raise ValueError("Human mask dimension is not compactible to image dimension, or according to the required shape. \
                             Expected shape is (1,{},{}) but got {}. {}".format(m_height, m_width, human_instance_masks[0]. shape,extra_debug_info))
        
        max_num_human_instance_masks = max(len(d) for d in human_instance_masks)
        if max_num_human_instance_masks > 0:
            human_instance_masks_padded = torch.ones((batch_size,max_num_human_instance_masks,m_height, m_width))*-1 # canvas of -1s

            for idx, im in enumerate(human_instance_masks):
                if im.shape[0]>0:
                    human_instance_masks_padded[idx,:im.shape[0],:,:] = im
                
                else:
                    # TODO: clean up this code properly.
                    raise ValueError("Current human instane mask is invalid. {}".format(extra_debug_info))
        
        else:
            human_instance_masks_padded = torch.ones((batch_size,max_num_human_instance_masks,m_height, m_width))*-1 # canvas of -1s
            raise ValueError("Empty human instance mask. {}".format(extra_debug_info))
        

        # Human instance bounding boxes and classes
        # =========================================
        human_instance_bbs_n_classes = [d['human_instance_bbs_n_classes'] for d in data]
        if not len(human_instance_bbs_n_classes) > 0:
            raise ValueError("No human instance segmentation annotation present! Please check `human_instance_bbs_n_classes` arguement. {}".format(extra_debug_info))
        
        if len(human_instance_bbs_n_classes) != len(human_instance_masks):
            raise ValueError("The number of human instance annotations ({}) do not match the number \
                             of human instance masks ({}). {}".format(len(human_instance_bbs_n_classes), len(human_instance_masks), extra_debug_info))
        
        max_num_human_instance_bbs_n_classes= max(len(d) for d in human_instance_bbs_n_classes)
        if max_num_human_instance_bbs_n_classes > 0:
            human_instance_bbs_n_classes_padded = torch.ones((batch_size,max_num_human_instance_bbs_n_classes,6))*-1 # canvas of -1s

            for idx, bb in enumerate(human_instance_bbs_n_classes):
                if bb.shape[0]>0:
                    human_instance_bbs_n_classes_padded[idx,:bb.shape[0],:] = bb
                
                else:
                    # TODO: clean up this code properly.
                    raise ValueError("Current human instance bounding box is invalid. {}".format(extra_debug_info))
        else:
            human_instance_bbs_n_classes_padded = torch.ones((batch_size,1,6))*-1 # dummy data of -1s
            raise ValueError("There are no bounding boxes for the human instance segmentation masks. {}".format(extra_debug_info))
        

        sample = {"images":images,
                  "images_path": images_path,
                  "part_instance_masks": part_instance_masks_padded,
                  "part_instance_bbs_n_classes": part_instance_bbs_n_classes_padded,
                  "human_instance_masks": human_instance_masks_padded,
                  "human_instance_bbs_n_classes": human_instance_bbs_n_classes_padded
                  }
        
        return sample
    

    def file_loc_debug_info(self,images_path:str=None):
        """Some text for extra debug information"""
        text = "Problem file can be found in the `Category_ids`, `Human_ids` or `Instance_ids` directory. \
            that correcponds to one of the files in this current file list {}. Try using a bactch size of 1 for debugging".format(images_path)
        return text
        
