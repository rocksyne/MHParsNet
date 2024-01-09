# 3rd party modules
import torch
import cv2 as cv
import numpy as np

# Application modules
from utilities.helper_utilities import GenerateImageData
from utilities.helper_utilities import Visualization
from utilities.utils import show_result_ins
torch.autograd.set_detect_anomaly(True)

# some global variables
computing_device = "cuda"
img_path = "data/sample_images/aau3.jpg"
model_path = "data/outputs/models/Entire_Model_MHP_20231027_143017.pt"


# class labels
humans_class_list = ("Human")
parts_class_list  = ("Cap/hat", "Helmet", "Face", "Hair", "Left-arm", "Right-arm", "Left-hand", "Right-hand",
                    "Protector", "Bikini/bra", "Jacket/windbreaker/hoodie", "Tee-shirt", "Polo-shirt", 
                    "Sweater", "Singlet", "Torso-skin", "Pants", "Shorts/swim-shorts",
                    "Skirt", "Stockings", "Socks", "Left-boot", "Right-boot", "Left-shoe", "Right-shoe",
                    "Left-highheel", "Right-highheel", "Left-sandal", "Right-sandal", "Left-leg", "Right-leg", "Left-foot", "Right-foot", "Coat",
                    "Dress", "Robe", "Jumpsuit", "Other-full-body-clothes", "Headwear", "Backpack", "Ball", "Bats", "Belt", "Bottle", "Carrybag",
                    "Cases", "Sunglasses", "Eyewear", "Glove", "Scarf", "Umbrella", "Wallet/purse", "Watch", "Wristband", "Tie",
                    "Other-accessary", "Other-upper-body-clothes", "Other-lower-body-clothes")

data = GenerateImageData(dimension=(512,512))
visuals = Visualization()
model = torch.load(model_path)

print('[INFO] Total params: {} numbers'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))

model = model.to(computing_device)
original, normalized, meta_data = data.get_data(img_path)

with torch.no_grad():
    parts_seg_result, human_seg_result = model(normalized.to(computing_device),operation_mode='evaluate',img_meta=[meta_data]) #normalized shape is [1,3,H,W]

original = original.numpy().transpose(1,2,0)
original = cv.cvtColor(original, cv.COLOR_BGR2RGB) # convert BGR to RGB

# show results 
part_img_show = show_result_ins(img = original,result = parts_seg_result,class_list = parts_class_list)
human_img_show = show_result_ins(img = original,result = human_seg_result,class_list = humans_class_list)

# append both images, side by side
aligned_image = np.concatenate((original,human_img_show, part_img_show ), axis=1)

cv.imwrite("test_image.png",aligned_image)

cv.imshow("watch windows",aligned_image)
if cv.waitKey(0) & 0xFF == ord('q'):
    cv.destroyAllWindows()

