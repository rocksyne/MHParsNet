# MHParsNet

## Results
### Seqmentation Mask Quality Evaluation </u>
![Figure 2. Application algorithm](segmentation_result.png)
Fig. 1. Segmentation result of MHParsNet on the validation set of the MHPV2 dataset. In the body part semantic mask
generation, MHParsNet misses the sunglasses object, which is reflected in the instance segmentation mask. It was however able
to detect the smaller wristwatch object.

The figure shows how the quality of our human parsing mask compares against the ground truth data of the MHPV2 dataset. In the prediction of the body part semantic (category) mask, MHParsNet misses the prediction of sunglasses. The reason for this is not the inability to detect small objects, but rather a false negative prediction. MHParsNet is capable detecting small objects such as the detecting of the watch on the wrist of the woman in the original image.
