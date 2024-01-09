# -*- coding: utf-8 -*-
import os
import os.path
import numpy as np
import random
import h5py
import torch
import torch.utils.data as udata
import PIL.Image as Image
from numpy.random import RandomState
import scipy.io as sio
import PIL
from PIL import Image
from .build_gemotry import initialization, build_gemotry

param = initialization()
ray_trafo, FBPOper, op_norm = build_gemotry(param)


def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max) 
    data = (data - data_min) / (data_max - data_min)
    data = data.astype(np.float32)
    data = data*255.0
    data = np.transpose(np.expand_dims(data, 2), (2, 0, 1))
    return data

class MARSPTrainDataset(udata.Dataset):
    def __init__(self, dir, mask, trainproj):
        super().__init__()
        self.dir = dir
        self.train_mask = mask
        self.txtdir = os.path.join(self.dir, 'train_640geo_dir.txt')
        self.mat_files = open(self.txtdir, 'r').readlines()
        self.file_num = len(self.mat_files)
        self.rand_state = RandomState(66)
        self.proj = trainproj  # sparse-view angles
    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        gt_dir = self.mat_files[idx]
        random_mask = random.randint(0, 89)  # include 89
        file_dir = gt_dir[:-6]
        data_file = file_dir + str(random_mask) + '.h5'
        abs_dir = os.path.join(self.dir, 'train_640geo/', data_file)
        gt_absdir = os.path.join(self.dir,'train_640geo/', gt_dir[:-1])
        gt_file = h5py.File(gt_absdir, 'r')
        Xgt = gt_file['image'][()]
        gt_file.close()
        file = h5py.File(abs_dir, 'r')
        Xma= file['ma_CT'][()]
        Sma = file['ma_sinogram'][()]
        XLI =file['LI_CT'][()]
        SLI = file['LI_sinogram'][()]
        TrMAR = file['metal_trace'][()]
        file.close()
        Np, Nb = Sma.shape  # Np:full-view projections, Nb: detectors
        D = np.zeros((Np, Nb), dtype=float)  # downsampling matrix
        factor = Np//self.proj 
        D[::factor,:] = 1 # if sample,1; 
        DI =  1 - D 
        Smasp = D * Sma 
        SLIsp = D * SLI
        Xmasp = FBPOper(Smasp)
        XLIsp = FBPOper(SLIsp)
        Sgt = np.asarray(ray_trafo(Xgt))
        M512 = self.train_mask[:,:,random_mask]
        M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
        Xma = normalize(Xmasp, image_get_minmax())  # *255
        Xgt = normalize(Xgt, image_get_minmax())
        XLI = normalize(XLIsp, image_get_minmax())
        Sma = normalize(Smasp, proj_get_minmax())
        Sgt = normalize(Sgt, proj_get_minmax())
        SLI = normalize(SLIsp, proj_get_minmax())   
        TrI_bool = np.logical_or(TrMAR, DI) # union, representing region needs to be restoration including TrMAR and Sparse-View
        TrI = np.zeros((Np, Nb), dtype=float)
        TrI[TrI_bool==True]=1
        TrDC = 1 -TrI # data consistency mask
        Tr = TrDC.astype(np.float32)
        Tr = np.transpose(np.expand_dims(Tr, 2), (2, 0, 1))
        
        TrI = TrI.astype(np.float32)
        TrI = np.transpose(np.expand_dims(TrI, 2), (2, 0, 1))
        
        TrMAR = TrMAR.astype(np.float32)
        TrMAR = np.transpose(np.expand_dims(TrMAR, 2), (2, 0, 1))
        
        DI = DI.astype(np.float32)
        DI = np.transpose(np.expand_dims(DI, 2), (2, 0, 1))
        
        
        Mask = M.astype(np.float32)
        Mask = np.transpose(np.expand_dims(Mask, 2), (2, 0, 1))
        return torch.Tensor(Xma), torch.Tensor(XLI), torch.Tensor(Xgt), torch.Tensor(Mask), \
               torch.Tensor(Sma), torch.Tensor(SLI), torch.Tensor(Sgt), torch.Tensor(Tr), torch.Tensor(TrI), torch.Tensor(TrMAR), torch.Tensor(DI) 


