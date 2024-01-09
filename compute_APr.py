from utilities.metrics import APr
import numpy as np


categories = ["Cap/hat", "Helmet", "Face", "Hair", "Left-arm", "Right-arm", "Left-hand", "Right-hand",
                        "Protector", "Bikini/bra", "Jacket/windbreaker/hoodie", "Tee-shirt", "Polo-shirt", 
                        "Sweater", "Singlet", "Torso-skin", "Pants", "Shorts/swim-shorts",
                        "Skirt", "Stockings", "Socks", "Left-boot", "Right-boot", "Left-shoe", "Right-shoe",
                        "Left-highheel", "Right-highheel", "Left-sandal", "Right-sandal", "Left-leg", "Right-leg", "Left-foot", "Right-foot", "Coat",
                        "Dress", "Robe", "Jumpsuit", "Other-full-body-clothes", "Headwear", "Backpack", "Ball", "Bats", "Belt", "Bottle", "Carrybag",
                        "Cases", "Sunglasses", "Eyewear", "Glove", "Scarf", "Umbrella", "Wallet/purse", "Watch", "Wristband", "Tie",
                        "Other-accessary", "Other-upper-body-clothes", "Other-lower-body-clothes"]


instance_pred_folder = "data/outputs/instance_pred_folder"
instance_gt_folder = "/media/rockson/Data_drive/datasets/LV-MHP-v2/val/Instance_ids"
num_classes = len(categories)

met = APr(instance_pred_folder, instance_gt_folder, num_classes)

AP_map = met.compute_AP()
for thre in AP_map.keys():
    print('threshold: {:.2f}, AP^r: {:.4f}'.format(thre, AP_map[thre]))
    print('Mean AP^r: {}'.format(np.nanmean(np.array(list(AP_map.values())))))