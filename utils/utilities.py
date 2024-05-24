import os, sys, glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize
# import nibabel as nib
import random
from numpy.lib.arraypad import pad

from statistics import median
# from scipy.misc import imrotate 
## scipy.ndimage.interpolation.rotate 
from scipy import ndimage, misc
from skimage import exposure
from progressbar import ProgressBar
# from skimage.metrics import structural_similarity as ssim
from skimage.metrics import (normalized_root_mse, peak_signal_noise_ratio,
                             structural_similarity)
import torchvision.utils as vutils
import multiprocessing.dummy as multiprocessing
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
###########################################################################
###########################################################################
###########################################################################
def pad_array(arr, sizemin):
    height, width = arr.shape
    if height < sizemin or width < sizemin:
        new_height = max(height, sizemin)
        new_width = max(width, sizemin)
        padded_arr = np.zeros((new_height, new_width))
        padded_arr[:height, :width] = arr
        return padded_arr
    else:
        return arr
###########################################################################
##### Data Loader Classes  ################################################
###########################################################################
class DataLoaderClass():
    """Motion Correction Dataset"""
    def __init__(self, input_list, transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.transform = transform

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## ----------------------------------------------------------
        file_name, file_extension = os.path.splitext(img_name_in_)
        # path_name = os.path.dirname(os.path.abspath(img_name_in_))
        file_name = file_name[:-4]
        ## ----------------------------------------------------------
        ## print(img_name_in_)
        image_in_ = nib.load(img_name_in_).get_fdata()
        image_in_ = image_in_/image_in_.max()
        label = np.array(file_name[-1], np.float32)
        if self.transform:
            image_in_ = self.transform(image_in_)
            label = self.transform(label)
        return torch.Tensor(image_in_), torch.Tensor(label)
    
class DataLoaderClassPNG():
    """Motion Correction Dataset"""
    def __init__(self, input_list, transform=None):
        """
        Args:
            input list (string): Path to the list of files;
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.files_in_ = list(input_list)
        self.transform = transform

    def __len__(self):
        return len(self.files_in_)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name_in_ = os.path.join(self.files_in_[idx])
        ## ----------------------------------------------------------
        file_name, file_extension = os.path.splitext(img_name_in_)
        # path_name = os.path.dirname(os.path.abspath(img_name_in_))
        file_name = file_name[:-4]
        ## ----------------------------------------------------------
        ## print(img_name_in_)
        image_in_ = cv2.imread(img_name_in_, cv2.IMREAD_GRAYSCALE)
        image_in_ = pad_array(image_in_, 64)
        
        image_in_ = image_in_/image_in_.max()
        label = np.array(file_name[-1], np.float32)
        if self.transform:
            image_in_ = self.transform(image_in_)
            label = self.transform(label)
        return torch.Tensor(image_in_), torch.Tensor(label)
    
###########################################################################
###########################################################################
###########################################################################
def tensorboard_classification(writer, inputs, labels, preds, epoch, section='train'):
    # Display the first image
    img = inputs[0, ...]
    writer.add_image('{}/input'.format(section),
                     vutils.make_grid(img,
                                      normalize=True,
                                      scale_each=True),
                     epoch)
    # Display the ground truth label
    writer.add_text('{}/Ground Truth Label'.format(section),
                    str(labels[0].item()),
                    epoch)
    # Display the predicted label
    writer.add_text('{}/Predicted Label'.format(section),
                    str(preds[0].item()),
                    epoch)



class CSVDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.data = self.data.drop(['filename'], axis=1) 
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        dfarray = self.data.to_numpy()
        image_in_ = dfarray[idx,1:]
        label = np.array(dfarray[idx,0])

        return torch.from_numpy(image_in_), torch.from_numpy(label)
