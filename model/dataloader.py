import os
import gc
import sys
from PIL import Image
import cv2
import math, random
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import timm

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
import albumentations as A

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import tifffile as tiff
import pandas as pd
import cv2
import torch
from torch.utils.data import Dataset
from utils import normalize_sentinel2_image


import albumentations as A
import cv2
def list_checkpoints(folder, extension='.pt'):
    checkpoint_paths = []
    for file_name in os.listdir(folder):
        if file_name.endswith(extension):
            checkpoint_paths.append(os.path.join(folder, file_name))
    return checkpoint_paths


    
def transformer(p=0.95):
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomRotate90(),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9, 
                         border_mode=cv2.BORDER_REFLECT)
       ,A.CoarseDropout(max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8, p=0.75),    
    ], p=p)

    

class OaDataset(Dataset):
    def __init__(self, root, df, imsz=(224, 224), transform=None):
        self.root = root
        self.df = df
        self.transform = transform
        self.imsz = imsz

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        t = self.df.iloc[idx]
        img_path = f'{self.root}train_images/{t["File_Name"]}'
        image = normalize_sentinel2_image(img_path)

        image = cv2.resize(image, self.imsz)


        if self.transform is not None:
            image = self.transform(image=image)['image']

        image = image.transpose(2, 0, 1)


        OaD = t["Value"]

        return image, OaD


class OaDPrediction(Dataset):
    def __init__(self, img_dir, phase='test', imsz=(224, 224), transform=None):
        """
        Args:
            img_dir (str): Root directory of the dataset.
            phase (str): Indicates whether the data is for training, validation, or testing.
            imsz (tuple): Target image size (width, height).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_dir = img_dir
        self.transform = transfor
        # Get a list of all image files in the directory
        self.img_files = [f for f in os.listdir(self.img_dir) if os.path.isfile(os.path.join(self.img_dir, f))]
    
    def __len__(self):
        # The length of the dataset is the number of image files
        return len(self.img_files)
    
    def __getitem__(self, idx):
        # Get the filename of the image
        img_filename = self.img_files[idx]
        
        # Construct the full path to the image
        img_path = os.path.join(self.img_dir, img_filename)
        
        # Read the image using tifffile
        
        image = normalize_sentinel2_image(img_path)
        image = cv2.resize(image, self.imsz)
        
        # Rearrange the image dimensions (channels, height, width)
        image = image.transpose(2, 0, 1)
        
        return image,img_filename  #, label (if you need to return a label too)