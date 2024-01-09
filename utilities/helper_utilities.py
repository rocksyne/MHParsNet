"""
	Author: Rockson Ayeman (rockson.agyeman@aau.at, rocksyne@gmail.com)
			Bernhard Rinner (bernhard.rinner@aau.at)

	At:     Pervasive Computing Group (https://nes.aau.at/?page_id=6065)
			University of Klagenfurt, 9020 Klagenfurt, Austria

	Date:   27.07.2022 (First authored date)

	Documentation:
	--------------------------------------------------------------------------------
	Utility functions for human parsing / segmentation research 

	Methods argument syntx: <variable name>:<data type>  = <expression>
							 [a,b,..n] represent possible argument values

	References:
		SaveBestModel(): https://debuggercafe.com/saving-and-loading-the-best-model-in-pytorch/

	TODO:
		pass
		
"""

# native modules
import sys, os
import json
from typing import Any

# 3rd pathy modules
import numpy as np
import torch
import pathlib
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.pylab import plt
import numpy as np
import copy
from torch.utils.data import Dataset
from torchvision.transforms import Normalize

from .utils import resize_image
from .utils import get_data_from_txt_file
from .utils import get_norm_values_from_file
from .utils import get_lr



# print numpy arrays in full without truncation
np.set_printoptions(threshold=sys.maxsize) # 
parent_directory = pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent


class SavePlots:
	def __init__(self,
	      		dir = os.path.join(pathlib.Path(os.path.dirname(os.path.realpath(__file__))).parent,"data","outputs","plots"),
				plot_name:str="train_VRS_val_loss.png", 
				plot_figure_size:tuple = (10,7),
				save_json_also:bool = True,
				overwrite:bool=True,
				number_of_plots:int=None,
				plot_title:str="Some Generic Title",
				x_axis_label:str="X axis label",
				y_axis_label:str="Y axis label"):
		"""
		Save plots during training.

		Args.:	<str>dir: path to where plots should be saved
				<str>plot_name: the name the plot should be saved as
				<tuple>plot_figure_size: the dimensin of the figure
				<bool>save_json_also: bool to save the corresponding JSON encoded details
				<bool>over_write: overwrite file or create new ones
		
		Return:	None
		"""
		self.dir = dir
		self.plot_name = plot_name
		self.plot_figure_size = plot_figure_size
		self.save_json_also = save_json_also
		self.overwrite = overwrite
		self.number_of_plots = number_of_plots
		self.plot_title = plot_title
		self.x_axis_label = x_axis_label
		self.y_axis_label = y_axis_label

		# data elements
		self.y_axis_data = {}
		self.legends = []

		# initailize lists
		for i in range(self.number_of_plots):
			self.y_axis_data[i] = []

		if not os.path.exists(self.dir):
			raise ValueError("Invalid destnation path. Path provided is `{}`".format(self.dir))
		
		if not isinstance(number_of_plots,int):
			raise ValueError("Please provide how many plots are going to be created. Provided number is {}.".format(number_of_plots))

	def __call__(self,y_axis:list,legend:list):
				
		if not isinstance(y_axis,list) or not isinstance(legend,list):
			raise ValueError("y_axis and legend must be of type list. Provided types are {} and {}".format(type(y_axis),type(legend)))
		
		if len(y_axis) != self.number_of_plots:
			raise ValueError("length of y_axis and `number_of_plots` must match. Provided lengths are {} and {}".format(self.number_of_plots,len(y_axis)))
		
		if len(y_axis) != len(legend):
			raise ValueError("y_axis and legend must have the same length. Provided lengths are {} and {}".format(len(y_axis),len(legend)))


		plt.figure(figsize=self.plot_figure_size)
		plt.title(self.plot_title)
		plt.xlabel(self.x_axis_label)
		plt.ylabel(self.y_axis_label)

		for i in range(len(y_axis)):
			self.y_axis_data[i].append(y_axis[i])
			plt.plot(self.y_axis_data[i], label=legend[i])

		# plt.xticks(arange(0, 21, 2)) # Set the tick locations
		plt.legend(loc='best')
		plt.savefig(os.path.join(self.dir,self.plot_name))
		plt.close() # https://heitorpb.github.io/bla/2020/03/18/close-matplotlib-figures/

		if self.save_json_also:
			self._save_json_also(self.y_axis_data,legend)

	
	def _save_json_also(self,y_axis_data, legend):
		""" 
		Save raw plot values as json files
		"""
		# use legend list to create dictionary keys
		data = {}

		for index, key in enumerate(legend,start=0):
			data[key] = y_axis_data[index]
		
		file_name = self.plot_name.split(".")[0]
		file_name += ".json"

		if self.overwrite is False:
			file_name = file_name if self.over_write==True else "epoch_{}_{}".format(str(self.epochs[-1]+1),file_name)

		with open(os.path.join(self.dir,file_name), 'w') as f:
			json.dump(data, f) 



		
		



# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
class EarlyStopping:

	def __init__(self, 
	      		patience:int = 5, 
				verbose:bool = False, 
				delta:int = 0, 
				output_path:str = None,
				model_name:str = "best_model.pth",
				trace_func:print = print):
		"""
		Args:
			patience (int): How long to wait after last time validation loss improved. Default: 7
			verbose (bool): If True, prints a message for each validation loss improvement. Default: False
			delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
			path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
			trace_func (function): trace print function. Default: print            
		"""
		self.patience = patience
		self.verbose = verbose
		self.counter = 0
		self.best_score = None
		self.early_stop = False
		self.val_loss_min = np.Inf
		self.delta = delta
		self.trace_func = trace_func
		self.model_name = model_name
		self.output_path = os.path.join(parent_directory,"data","outputs","models") if output_path is None else output_path


	def __call__(self, val_loss, model, epoch, optimizer):

		score = -val_loss

		if self.best_score is None:
			self.best_score = score
			self.save_checkpoint(val_loss, model, epoch, optimizer)

		elif score < self.best_score + self.delta:
			self.counter += 1
			print("")
			self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
			if self.counter >= self.patience:
				self.early_stop = True

		else:
			self.best_score = score
			self.save_checkpoint(val_loss, model, epoch, optimizer)
			self.counter = 0


	def save_checkpoint(self, val_loss, model, epoch, optimizer):
		'''Saves model when validation loss decrease.'''
		if self.verbose:
			self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
		
		# put model in eval mode
		model = model.eval()

		# save state dictionaries
		torch.save({'epoch': epoch+1,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()
					}, os.path.join(self.output_path,'Model_State_Dict_{}'.format(self.model_name)))
		
		# save the entire model
		torch.save(model, os.path.join(self.output_path,'Entire_Model_{}'.format(self.model_name)))

		self.val_loss_min = val_loss





class SaveModel(object):
	def __init__(self,
	      		destination_path:str=os.path.join(parent_directory,"data","outputs","models"),
	      		model_name:str="best_model.pth",
				over_write:bool=True):
		"""
		Save training model. This operation is traditionally a blocking process.
		But for optimization process (speed in training), we push this to a background
		(process in a separate thread).

		Args.: 	<str>dir: destination path. Default is data/outputs/models
				<str>model_name: name to be given to model, eg. best_model.pt
				<bool>overwrite: overwrite file or create new ones
		"""
		super(SaveModel, self).__init__()
		self.destination_path = destination_path
		self.model_name = model_name
		self.over_write = over_write

	#@background.task
	def __call__(self, model=None, epoch:int=None, optimizer:torch.optim=None):
		""" 
		Save the model, but do it as a background / non blocking process

		Args.: 	<model>model: network or model
				<int>epoch: current epoch of the training process
				<optimizer>optimizer: optimizer function
				<loss_fnc>loss_fnc: loss function
		"""
		# security checks
		if model is None or epoch is None or optimizer is None:
			raise ValueError("Invalid values for parameters.")
		
		# prepare model name
		model_name = self.model_name if self.over_write==True else "epoch_{}_{}".format(str(epoch+1),self.model_name)

		torch.save({
				'epoch': epoch+1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()
				}, os.path.join(self.destination_path,model_name))
		
		model = model.eval()
		torch.save(model, os.path.join(self.destination_path,'Entire_Model_{}'.format(model_name))) # save the whole mdel





class ModelStatistics(object):
	def __init__(self):
		"""
		This class is a utility tool to estimate the model size in memory,
		as well as calculating the number of model parameters. See the references below for details.
		https://discuss.pytorch.org/t/finding-model-size/130275
		Code is adopted from https://discuss.pytorch.org/t/finding-model-size/130275/2

		Args.:	<torch.Tensor>model: neural network model
		Return:	<list> num_params, model_size
		"""
		... # this means we have nothing to put here
	
	def __call__(self, model:torch.Tensor=None):
		param_size = 0
		buffer_size = 0

		for param in model.parameters():
			param_size += param.nelement() * param.element_size()

		for buffer in model.buffers():
			buffer_size += buffer.nelement() * buffer.element_size()

		# https://discuss.huggingface.co/t/how-to-get-model-size/11038/2
		total_params = sum(param.numel() for param in model.parameters())

		size_all_mb = (param_size + buffer_size) / 1024**2
		return '{:,}'.format(total_params), '{:.3f} MB'.format(size_all_mb)
	


class GenerateImageData(Dataset):

	def __init__(self,
	      		dimension: tuple = (544, 544),
	      		normalization_file_path:str = os.path.join(parent_directory,"utilities","normal_file.json"),
				interpolation:str = "bilinear"):
		""" 
		Return: <torch.Tensor>original_image: resized image of shape [3,H,W]
				<torch.Tensor>normalized_image: 
		"""
		super(GenerateImageData, self).__init__()
		self.dimension = dimension
		self.interpolation = interpolation
		norm_values = get_norm_values_from_file(normalization_file_path)
		self.normalize_image = Normalize(norm_values['means'], norm_values['stds'], inplace=False)
	
	def get_data(self,image_path):
		image_data = cv.imread(image_path,cv.IMREAD_COLOR) # (H,W,3) where 3 is GBR
		image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB) # convert BGR to RGB
		image_data = resize_image(image_data,size=self.dimension,interpolation=self.interpolation)

		original_image_shape = image_data.shape

		image_data = image_data.transpose(2,0,1) # [H,W,3] >> [3,H,W]
		image_data = torch.from_numpy(image_data) # convert to torch array
		original_image = copy.deepcopy(image_data) # make a copy before normalizing
		image_data = self.normalize_image(image_data.type(torch.float))
		image_data = image_data.unsqueeze(0) # add a dimention for the batch axis [N,C,H,W]


		# print("Original image shape: ",original_image_shape)
		meta_data = dict(img_shape=original_image_shape,
		     			scale_factor=1,
						ori_shape = original_image_shape)
		
		return original_image, image_data, meta_data
	

	def process_cam_data(self,image_data):
		image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB) # convert BGR to RGB
		image_data = resize_image(image_data,size=self.dimension,interpolation=self.interpolation)

		original_image_shape = image_data.shape

		image_data = image_data.transpose(2,0,1) # [H,W,3] >> [3,H,W]
		image_data = torch.from_numpy(image_data) # convert to torch array
		original_image = copy.deepcopy(image_data) # make a copy before normalizing
		image_data = self.normalize_image(image_data.type(torch.float))
		image_data = image_data.unsqueeze(0) # add a dimention for the batch axis [N,C,H,W]

		meta_data = dict(img_shape=original_image_shape,
		     			scale_factor=1,
						ori_shape = original_image_shape)
		
		return original_image, image_data, meta_data
	


class Visualization(object):
	def __init__(self):
		...

	def show_image(self,image:torch.Tensor, title:str="Display Image"):

		# some house cleaning
		if isinstance(image,torch.Tensor):
			image = image.numpy().transpose(1,2,0) # shape [H,W,C]
			image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # RGB color space

		cv.imshow(title, image)
		k = cv.waitKey(0)
		if k == 27:  # close on ESC key
			cv.destroyAllWindows()


	def save_image(self,image:torch.Tensor, image_name:str="saved_image.jpg"):
		# some house cleaning
		if isinstance(image,torch.Tensor):
			image = image.numpy().transpose(1,2,0) # shape [H,W,C]
			image = cv.cvtColor(image, cv.COLOR_RGB2BGR) # RGB color space
		cv.imwrite(image_name, image)


