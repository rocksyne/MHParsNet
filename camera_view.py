# system modules
import os

# 3rd party modules
import torch
import numpy as np
import cv2 as cv
from time import time, sleep

# Application modules
from modules.mhparsnet import MPHParsNet
from utilities.helper_utilities import GenerateImageData
from utilities.webcam import WebcamVideoStream
from utilities.helper_utilities import Visualization
from utilities.utils import show_result_ins
torch.autograd.set_detect_anomaly(True)

# some global variables
computing_device = "cuda"
save_video = False
saved_model_path = "/media/rockson/Data_drive/Research/MHParsNetRework/data/outputs/models/Entire_Model_MHP_20231027_143017.pt"

# class labels
humans_class_list = ("Human")
parts_class_list = ("Cap/hat", "Helmet", "Face", "Hair", "Left-arm", "Right-arm", "Left-hand", "Right-hand",
                    "Protector", "Bikini/bra", "Jacket/windbreaker/hoodie", "Tee-shirt", "Polo-shirt", 
                    "Sweater", "Singlet", "Torso-skin", "Pants", "Shorts/swim-shorts",
                    "Skirt", "Stockings", "Socks", "Left-boot", "Right-boot", "Left-shoe", "Right-shoe",
                    "Left-highheel", "Right-highheel", "Left-sandal", "Right-sandal", "Left-leg", "Right-leg", "Left-foot", "Right-foot", "Coat",
                    "Dress", "Robe", "Jumpsuit", "Other-full-body-clothes", "Headwear", "Backpack", "Ball", "Bats", "Belt", "Bottle", "Carrybag",
                    "Cases", "Sunglasses", "Eyewear", "Glove", "Scarf", "Umbrella", "Wallet/purse", "Watch", "Wristband", "Tie",
                    "Other-accessary", "Other-upper-body-clothes", "Other-lower-body-clothes")



data = GenerateImageData(dimension=(512,512))
visuals = Visualization()
camera = WebcamVideoStream(src=0).start() # read the usb camera
sleep(1.0) # get camera ready -- warm up

# load model
model = torch.load(saved_model_path)
model = model.to(computing_device)

# Define the codec and create VideoWriter object
out = cv.VideoWriter('smart_cam_output.mp4', cv.VideoWriter_fourcc(*'MP4V'),10, (512,512))


# FPS initializers
frame_count  = 0
start_time = time()

while True:
    # check if a frame has been read
    if camera.grabbed is False:
        break

    # Capture frame-by-frame and preprocess
    frame = camera.read()
    frame = cv.flip(frame,1)
    original, normalized, meta_data = data.process_cam_data(frame)

    with torch.no_grad():
        parts_seg_result, human_seg_result = model(normalized.to(computing_device),operation_mode='evaluate',img_meta=[meta_data]) #normalized shape is [1,3,H,W]
    
    original = original.numpy().transpose(1,2,0)
    original = cv.cvtColor(original, cv.COLOR_BGR2RGB) # convert BGR to RGB

    # show results 
    part_img_show = show_result_ins(img = original,result = parts_seg_result,class_list = parts_class_list)
    human_img_show = show_result_ins(img = original,result = human_seg_result,class_list = humans_class_list)

    # append both images, side by side
    aligned_image = np.concatenate((part_img_show, human_img_show), axis=1)

    frame_count  += 1
    frame_rate = frame_count / (time() - start_time)

    cv.putText(aligned_image, f"Frame Rate: {frame_rate:.2f} FPS", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv.imshow("Frame", aligned_image)

    # save camera output as video
    if save_video:
        out.write(part_img_show)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

    # prevent number from overgrowing
    if frame_count >= 1200:
        frame_count  = 0
        start_time = time()

camera.stop()
out.release()
cv.destroyAllWindows()