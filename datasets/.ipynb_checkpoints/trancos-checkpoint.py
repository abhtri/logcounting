from torch.utils import data
import numpy as np
import torch
import sys
import os
import cv2
from skimage.io import imread
from skimage.transform import rescale, resize

from scipy.io import loadmat
sys.path.append('..')
import utils as ut
import torchvision.transforms.functional as FT



class Trancos(data.Dataset):
    def __init__(self, root="",split=None, 
                 transform_function=None):
        self.split = split
        
        self.n_classes = 2
        self.transform_function = transform_function
        
        ############################
        # self.path_base = "/home/tammy/LCFCN/datasets/TRANCOS_v3"
        self.path_base ="/floyd/input/logcounting"
        # self.path_base = "/mnt/datasets/public/issam/Trancos/"

        if split == "train":
            fname = self.path_base + "/image_sets/training.txt"

        elif split == "val":
            fname = self.path_base + "/image_sets/validation.txt"

        elif split == "test":
            fname = self.path_base + "/image_sets/test.txt"

        self.img_names = [name.replace(".jpg\n","") for name in ut.read_text(fname)]
        self.path = self.path_base + "/images/"
        self.path_dots = self.path_base + "/dots/"
        assert os.path.exists(self.path + self.img_names[0] + ".jpg")
        

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        name = self.img_names[index]

        # LOAD IMG, POINT, and ROI
        try:
            image = imread(self.path + name + ".jpg")
        except FileNotFoundError:
            image = imread(self.path + name)
        # image = resize(image, (image.shape[0], image.shape[1]),
        #                anti_aliasing=True)   

        points = imread(self.path_dots + name + ".png")
        try:
            points = points[:,:,:1].clip(0,1)
        except Exception:
            points = np.stack((points,)*3, axis=-1)
            points = points[:,:,:1].clip(0,1)
        # roi = loadmat(self.path + name + "mask.mat")["BW"][:,:,np.newaxis]
        roi = 1
            

        # LOAD IMG AND POINT
        image = image 
        # image = ut.shrink2roi(image, roi)
        # points = ut.shrink2roi(points, roi).astype("uint8")

        counts = torch.LongTensor(np.array([int(points.sum())]))   
        
        collection = list(map(FT.to_pil_image, [image, points]))
        
        if self.transform_function is not None:
            image, points = self.transform_function(collection)
            
        if (points == -1).all():
            pass
        else:
            assert int(points.sum()) == counts[0]
        
        return {"images":image, "points":points, 
                "counts":counts, "index":index,
                "image_path":self.path + name + ".jpg"}
