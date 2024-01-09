# system modules
import os
import datetime
import timeit

# 3rd party modules
import torch
import torch.optim as optim
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import warnings

# application modules
from utilities.MHPv2_manager import MHPv2
from utilities.MHPv2_manager import MHPv2DataCollator
from modules.mhparsnet import MPHParsNet
from utilities.utils import get_timestamp_as_str
from utilities.utils import clip_grads
from utilities.utils import get_lr 
from utilities.utils import set_lr 
from utilities.utils import gradinator
from utilities.utils import get_warmup_lr
from utilities.utils import get_model_size
from utilities.utils import clean_data_recieved_from_collator
from utilities.helper_utilities import EarlyStopping
from utilities.helper_utilities import SavePlots
from utilities.progress_bar import ProgressBar

torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore") 


# Modify global parameters
server_type = "slow" # choose server for training
data_dimension = (512,512) # size of the training image
epochs = 500 # unrealistic, but a safe number enough
scheduler_patience = 3 # learning rate scheduler patience counter
time_stamp = get_timestamp_as_str() # as function name implies
model_name = "MHP_{}.pt".format(time_stamp)
plot_name  = "MHP_{}.png".format(time_stamp)
initial_LR = 0.01
do_warm_up = True # use warm-up training strategy, or not
warmup_iterations = 1000

# lazy fix for switching between training servers
if server_type == "fast":
    dataset_dir = "/home/users/roagyeman/research/datasets/LV-MHP-v2"
    batch_size = 8
    workers = 18
    progress_bar_width = 1
    computing_device = "cuda" if torch.cuda.is_available() else "cpu"
    
elif server_type == "slow":
    dataset_dir = "/media/rockson/Data_drive/datasets/LV-MHP-v2-bkup"
    batch_size = 2
    workers = 8
    progress_bar_width = 1
    computing_device = "cuda" if torch.cuda.is_available() else "cpu"

else:
    raise ValueError("Invalid name for server")


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth'}

collate_fn = MHPv2DataCollator(image_dimension=data_dimension)
dataset_train = MHPv2(dimension=data_dimension, 
                        dataset_parent_dir=dataset_dir, 
                        dataset_split="train")

dataset_val  = MHPv2(dimension=data_dimension, 
                        dataset_parent_dir=dataset_dir, 
                        dataset_split="val")

training_data   = DataLoader(dataset_train, num_workers=workers, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
validation_data = DataLoader(dataset_val, num_workers=workers, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)

model = MPHParsNet(num_of_classes = 59,
               resnet_version = 'resnet34',
               pyramid_levels = [3,4,5,6,7],
               computing_device = computing_device)

model.load_state_dict(model_zoo.load_url(model_urls['resnet34'], model_dir='./data/weights'), strict=False) 

print('[INFO] Total params: {}'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))

# Freeze BN layers
for name, module in model.named_modules():
    if isinstance(module, nn.BatchNorm2d):
        module.eval()

model = model.to(computing_device)

optimizer = optim.SGD(model.parameters(), lr=initial_LR, momentum=0.9, weight_decay=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=scheduler_patience, verbose=True)
stop_early = EarlyStopping(patience=scheduler_patience+3, model_name=model_name)


save_total_loss_plot = SavePlots(plot_name = "total_MHP_loss.png",
                                          number_of_plots=2,
                                          plot_title = "Total Human Parsing Loss",
                                          x_axis_label = "Epochs",
                                          y_axis_label = "Losses")

print("")
print('[INFO] Total params: {} M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))
print("")
print("[INFO] Training started {}".format(datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")))
print("")

train_start_time = timeit.default_timer()
current_training_step = 0

for current_epoch in range(epochs):

    train_part_instance_loss = []
    train_part_category_loss = []
    train_part_total_loss = []
    train_human_instance_loss = []
    train_human_category_loss = []
    train_human_total_loss = []
    train_total_loss = []

    val_part_instance_loss = []
    val_part_category_loss = []
    val_part_total_loss = []
    val_human_instance_loss = []
    val_human_category_loss = []
    val_human_total_loss = []
    val_total_loss = []

    train_progress_bar = ProgressBar(prefix_text="Train",
                                     target=len(training_data), 
                                     epoch=current_epoch , 
                                     num_epochs=epochs, 
                                     width=progress_bar_width,
                                     always_stateful=True)

    model.train()
    for training_step, data in enumerate(training_data):

        # perform learning rate warm up for a fixed number of iterations
        # I prefer one or two epochs, generally
        if (current_training_step <= warmup_iterations) and (do_warm_up is True):
            warm_lr = get_warmup_lr(current_training_step, warmup_iterations,initial_LR,0.01,'linear')
            set_lr(optimizer, warm_lr)
            current_training_step +=1
                
        elif (current_training_step >= warmup_iterations) and (do_warm_up is False):
            set_lr(optimizer, initial_LR) 
            do_warm_up = False # ignore warm up from now on
        
        
        # TODO: Optimize codes in future
        images = data['images']
        images = gradinator(images.to(computing_device))

        # Prepare / clean the parts data recieved from collator
        parts_bbs_n_labels = data['part_instance_bbs_n_classes']
        parts_instance_masks = data['part_instance_masks']
        parts_gt_bboxes, parts_gt_labels, parts_gt_masks = clean_data_recieved_from_collator(parts_bbs_n_labels,parts_instance_masks,computing_device)

        # Prepare / clean the human instance data recieved from collator
        human_bbs_n_labels = data['human_instance_bbs_n_classes']
        human_instance_masks = data['human_instance_masks']
        human_gt_bboxes, human_gt_labels, human_gt_masks = clean_data_recieved_from_collator(human_bbs_n_labels,human_instance_masks,computing_device)

        # Zero the gradients for every batch!
        optimizer.zero_grad()

        # make predictions and estimate losses for part and human instances
        parts_loss, human_loss = model(image_inputs=images,
                                       parts_gt_bboxes=parts_gt_bboxes,
                                       parts_gt_labels=parts_gt_labels,
                                       parts_gt_masks=parts_gt_masks,
                                       human_gt_bboxes=human_gt_bboxes,
                                       human_gt_labels=human_gt_labels,
                                       human_gt_masks=human_gt_masks,
                                       operation_mode='train')

        # get part losses
        parts_total_loss = parts_loss['loss_ins'] + parts_loss['loss_cate']
        parts_instance_loss = parts_loss['loss_ins'].cpu().item()
        parts_category_loss = parts_loss['loss_cate'].cpu().item()
        parts_total_loss_value  = parts_total_loss.cpu().item()

        # get human losses
        human_total_loss = human_loss['loss_ins'] + human_loss['loss_cate']
        human_instance_loss = human_loss['loss_ins'].cpu().item()
        human_category_loss = human_loss['loss_cate'].cpu().item()
        human_total_loss_value  = human_total_loss.cpu().item()

        # total losses
        total_loss = parts_total_loss + human_total_loss
        total_loss_value = total_loss.cpu().item()
        
        # back propagate
        total_loss.backward()

        # Adjust learning weights if they are not 
        if torch.isfinite(total_loss).item():
            grad_norm = clip_grads(model.parameters())
            optimizer.step()
        else:
            NotImplementedError("loss type error!can't backward!")
        
        # save losses
        train_part_instance_loss.append(parts_instance_loss)
        train_part_category_loss.append(parts_category_loss)
        train_part_total_loss.append(parts_total_loss_value)
        train_human_instance_loss.append(human_instance_loss)
        train_human_category_loss.append(human_category_loss)
        train_human_total_loss.append(human_total_loss_value)
        train_total_loss.append(total_loss_value)
        
        # update progress bar with selected metrics. Loss only for now
        values = [("P. inst. loss",float(np.mean(train_part_instance_loss))),
                  ("P. cat. loss",float(np.mean(train_part_category_loss))),
                  ("P. total loss",float(np.mean(train_part_total_loss))),
                  ("H. inst. loss",float(np.mean(train_human_instance_loss))),
                  ("H. cat. loss",float(np.mean(train_human_category_loss))),
                  ("H. total loss",float(np.mean(train_human_total_loss))),
                  ("Total loss",float(np.mean(train_total_loss))),
                  ("LR",get_lr(optimizer))]

        train_progress_bar.update(current=training_step, values=values)


    # do validation data
    with torch.no_grad(): 

        # progress bar matters
        val_progress_bar = ProgressBar(prefix_text="Val  ",
                                     target=len(validation_data), 
                                     epoch=current_epoch , 
                                     num_epochs=epochs, 
                                     width=progress_bar_width,
                                     show_epoch_progress = False,
                                     always_stateful=True)
        
        model.eval()
        for validation_step, data in enumerate(validation_data, start=1):
            # TODO: Optimize codes in future
            images = data['images']
            images = gradinator(images.to(computing_device))

            # Prepare / clean the parts data recieved from collator
            parts_bbs_n_labels = data['part_instance_bbs_n_classes']
            parts_instance_masks = data['part_instance_masks']
            parts_gt_bboxes, parts_gt_labels, parts_gt_masks = clean_data_recieved_from_collator(parts_bbs_n_labels,parts_instance_masks,computing_device)

            # Prepare / clean the human instance data recieved from collator
            human_bbs_n_labels = data['human_instance_bbs_n_classes']
            human_instance_masks = data['human_instance_masks']
            human_gt_bboxes, human_gt_labels, human_gt_masks = clean_data_recieved_from_collator(human_bbs_n_labels,human_instance_masks,computing_device)

            # make prediction and get the loss
            parts_loss, human_loss = model(image_inputs=images,
                                        parts_gt_bboxes=parts_gt_bboxes,
                                        parts_gt_labels=parts_gt_labels,
                                        parts_gt_masks=parts_gt_masks,
                                        human_gt_bboxes=human_gt_bboxes,
                                        human_gt_labels=human_gt_labels,
                                        human_gt_masks=human_gt_masks,
                                        operation_mode='train')

            # get part losses
            parts_total_loss = parts_loss['loss_ins'] + parts_loss['loss_cate']
            parts_instance_loss = parts_loss['loss_ins'].cpu().item()
            parts_category_loss = parts_loss['loss_cate'].cpu().item()
            parts_total_loss_value  = parts_total_loss.cpu().item()

            # get human losses
            human_total_loss = human_loss['loss_ins'] + human_loss['loss_cate']
            human_instance_loss = human_loss['loss_ins'].cpu().item()
            human_category_loss = human_loss['loss_cate'].cpu().item()
            human_total_loss_value  = human_total_loss.cpu().item()

            # total losses
            total_loss = parts_total_loss + human_total_loss
            total_loss_value = total_loss.cpu().item()

            # save validation losses
            val_part_instance_loss.append(parts_instance_loss)
            val_part_category_loss.append(parts_category_loss)
            val_part_total_loss.append(parts_total_loss_value)
            val_human_instance_loss.append(human_instance_loss)
            val_human_category_loss.append(human_category_loss)
            val_human_total_loss.append(human_total_loss_value)
            val_total_loss.append(total_loss_value)
            
            # update progress bar with selected metrics. Loss only for now
            values = [("P. inst. loss",float(np.mean(val_part_instance_loss))),
                    ("P. cat. loss",float(np.mean(val_part_category_loss))),
                    ("P. total loss",float(np.mean(val_part_total_loss))),
                    ("H. inst. loss",float(np.mean(val_human_instance_loss))),
                    ("H. cat. loss",float(np.mean(val_human_category_loss))),
                    ("H. total loss",float(np.mean(val_human_total_loss))),
                    ("Total loss",float(np.mean(val_total_loss)))]
    
            val_progress_bar.update(current=validation_step, values=values)
    
    print("") # properly format the validation progress bar on the next line

    # check early stopping and save results / stop training accordingly
    stop_early(float(np.mean(val_total_loss)), model, current_epoch, optimizer)
    if stop_early.early_stop:
        break

    scheduler.step(np.mean(val_total_loss))
    
    save_total_loss_plot([np.mean(train_total_loss),
                        np.mean(val_total_loss)],
                       ["Train Total Loss",
                        "Val Total Loss"])
    
# end the timer for the trainging
time_elaspsed_in_sec = timeit.default_timer() - train_start_time
time_elapsed = datetime.timedelta(seconds=time_elaspsed_in_sec)
print("[Info] Total training time (HH:m:s) - ",time_elapsed)
print("") 
