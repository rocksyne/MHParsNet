"""
Doc.: Code for Multi Human Parsong Network (MHParsNet)
"""

import torch
import torch.nn as nn
from .model_modules import PyramidFeatures
from .model_modules import ResNetBackboneNetwork
from .parsing_head import ParsingHead
from .mask_head import MaskFeatHead


class MPHParsNet(nn.Module):
    
    def __init__(self,
                 num_of_classes:int=59,
                 resnet_version:str='resnet34',
                 pyramid_levels:list=[3,4,5,6,7],
                 computing_device='cpu'):
        
        super(MPHParsNet, self).__init__()
        self.num_of_classes = num_of_classes
        self.computing_device = computing_device

        self.resnet_back_bone = ResNetBackboneNetwork(resnet_version=resnet_version) # dictionary element
        self.pyramid_features = PyramidFeatures(C3_size=128, C4_size=256, C5_size=512, feature_size=256, pyramid_levels=pyramid_levels)
        
        self.parts_bbox_head = ParsingHead(num_classes=num_of_classes, #  total number of part instances (including the background)
                                in_channels=256,
                                seg_feat_channels=256,
                                stacked_convs=2,
                                strides=[8, 8, 16, 32, 32],
                                scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                                num_grids=[40, 36, 24, 16, 12],
                                ins_out_channels=128)
        self.parts_bbox_head.init_weights()

        # start level and end level determine the number of convolution stacks
        # start_level 0 is always the default, and end_level determines how many 
        # convolution stacks are used. In this work, we use two stacks 0 ~ 1
        self.parts_mask_feat_head = MaskFeatHead(in_channels=256,
                            out_channels=128,
                            start_level=0,
                            end_level=1,
                            num_classes=128)
        

        self.human_bbox_head = ParsingHead(num_classes=2, # either human exists or not
                                in_channels=256,
                                seg_feat_channels=256,
                                stacked_convs=2,
                                strides=[8, 8, 16, 32, 32],
                                scale_ranges=((1, 56), (28, 112), (56, 224), (112, 448), (224, 896)),
                                num_grids=[40, 36, 24, 16, 12],
                                ins_out_channels=128)
        self.human_bbox_head.init_weights()
        
        # see self.parts_mask_feat_head
        self.human_mask_feat_head = MaskFeatHead(in_channels=256,
                                                  out_channels=128,
                                                  start_level=0,
                                                  end_level=1,
                                                  num_classes=128)
        
        
        self.test_configurations = dict(nms_pre=500,
                                        score_thr=0.1,
                                        mask_thr=0.5,
                                        update_thr=0.05,
                                        kernel='gaussian',  # gaussian/linear
                                        sigma=2.0,
                                        max_per_img=30)
        
        self.rescale = False


    def forward(self,
                image_inputs:torch.Tensor=None,
                img_meta:dict=None,
                parts_gt_bboxes:torch.Tensor=None,
                parts_gt_masks:torch.Tensor=None,
                parts_gt_labels:torch.Tensor=None,
                parts_gt_bboxes_ignore=None,
                human_gt_bboxes:torch.Tensor=None,
                human_gt_masks:torch.Tensor=None,
                human_gt_labels:torch.Tensor=None,
                human_gt_bboxes_ignore=None,
                operation_mode=None):
        """
        Args.:  <torch.Tensor> image_inputs. Shape [N,C,H,W]
                <torch.Tensor> gt_bounding_boxes
                <torch.Tensor> gt_masks
                <torch.Tensor> gt_labels
        """
        
        if operation_mode not in ["train","evaluate"]:
             raise ValueError("Invalid operation mode. Only `train` and `evaluate` modes are allowed. \
                              Provided operation mode is `{}`".format(operation_mode))
        
        if operation_mode == "train":
             if (image_inputs is None) or (parts_gt_bboxes is None) or (parts_gt_masks is None) or (parts_gt_labels is None):
                  raise ValueError("None value for one of the input arguements")
             
             if (human_gt_bboxes is None) or (human_gt_masks is None) or (human_gt_labels is None):
                  raise ValueError("None value for one of the input arguements")
        
        elif operation_mode == "evaluate":
             if (image_inputs is None) or (img_meta is None):
                  raise ValueError("There is data for `image_inputs` or `img`")
        
        # backbone output / output from the ResNet backbone 
        # resnet_back_bone_output = [c2,c3,c4,c5]
        # Deatailed architecture is at ../docs/fpn.webp
        resnet_back_bone_output = self.resnet_back_bone (image_inputs)

        # get the feature pyramid outputs
        # we use the resnet_back_bone_outputs c2, c3, c4, and c5, to generate feature pyramids
        # we shall use c2 generates M2 and P2, c3 generates M3 and P3,.... and on and on
        # Deatailed architecture is at ../docs/fpn.webp
        pyramid_features_output = self.pyramid_features(resnet_back_bone_output) # resnet_back_bone_output = [c2,c3,c4,c5]
        P2 = pyramid_features_output["P2"]
        P3 = pyramid_features_output["P3"]
        P4 = pyramid_features_output["P4"]
        P5 = pyramid_features_output["P5"]
        P6 = pyramid_features_output["P6"]
        P7 = pyramid_features_output["P7"]
        M2 = pyramid_features_output["M2"]

        x = tuple([P2, P3, P4, P5, P6])
        
        # return loss function in training mode
        # TODO: clean up code - too messy and repetetive. Maybe put in a method or function
        # Time constaraint so clean later after paper submission
        if operation_mode == "train":
                # parts losses
                parts_outs = self.parts_bbox_head(x)
                parts_mask_feat_pred = self.parts_mask_feat_head(x[self.parts_mask_feat_head.start_level:self.parts_mask_feat_head.end_level + 1])
                parts_loss_inputs = parts_outs + (parts_mask_feat_pred, parts_gt_bboxes, parts_gt_labels, parts_gt_masks, img_meta)
                parts_losses = self.parts_bbox_head.loss(*parts_loss_inputs, gt_bboxes_ignore=parts_gt_bboxes_ignore)
     
                # human losses
                human_outs = self.human_bbox_head(x)
                human_mask_feat_pred = self.human_mask_feat_head(x[self.human_mask_feat_head.start_level:self.human_mask_feat_head.end_level + 1])
                human_loss_inputs = human_outs + (human_mask_feat_pred, human_gt_bboxes, human_gt_labels, human_gt_masks, img_meta)
                human_losses = self.human_bbox_head.loss(*human_loss_inputs, gt_bboxes_ignore=human_gt_bboxes_ignore)
                
                return parts_losses, human_losses

        # return segmentation masks
        elif operation_mode == 'evaluate':
             # parts segmentation results
             parts_outs = self.parts_bbox_head(x,eval=True)
             parts_mask_feat_pred = self.parts_mask_feat_head(x[self.parts_mask_feat_head.start_level:self.parts_mask_feat_head.end_level + 1])
             parts_seg_inputs = parts_outs + (parts_mask_feat_pred, img_meta, self.test_configurations, self.rescale )
             parts_seg_result = self.parts_bbox_head.get_seg(*parts_seg_inputs)

             # human segmentation results
             human_outs = self.human_bbox_head(x,eval=True)
             human_mask_feat_pred = self.human_mask_feat_head(x[self.human_mask_feat_head.start_level:self.human_mask_feat_head.end_level + 1])
             human_seg_inputs = human_outs + (human_mask_feat_pred, img_meta, self.test_configurations, self.rescale )
             human_seg_result = self.human_bbox_head.get_seg(*human_seg_inputs)
             
             return parts_seg_result, human_seg_result