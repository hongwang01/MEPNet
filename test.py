import cv2
import os
import os.path
import argparse
import numpy as np
import torch
from math import ceil
import odl
from odl.contrib import torch as odl_torch
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr
import time
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
import PIL
from PIL import Image
from utils.build_gemotry import initialization, build_gemotry
from network.mepnet import MEPNet

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description="YU_Test")
parser.add_argument("--model_dir", type=str, default="pretrained_model/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="data/test/", help='path to test data')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--save_path", type=str, default="./test_results/", help='path to reconstructed data')
parser.add_argument('--num_channel', type=int, default=32, help='the number of dual channels')
parser.add_argument('--T', type=int, default=4, help='the number of ResBlocks in every ProxNet')
parser.add_argument('--S', type=int, default=10, help='the number of iterative stages ')
parser.add_argument('--eta1', type=float, default=1, help='stepsize')
parser.add_argument('--eta2', type=float, default=5, help='stepsize')
parser.add_argument('--alpha', type=float, default=0.5, help='initialization for weight factor')
parser.add_argument('--test_proj', type=int, default=160, help='the number of projection views')
opt = parser.parse_args()

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + " !  ---")


input_dir = opt.save_path + str(opt.test_proj) +'/input/'
out_dir = opt.save_path+ str(opt.test_proj) +'/EPNet/'
gt_dir = opt.save_path + str(opt.test_proj) +'/gt/' 

mkdir(input_dir)
mkdir(out_dir)
mkdir(gt_dir)


def normalized(X):
    maxX = np.max(X)
    minX = np.min(X)
    X = (X - minX) / (maxX - minX)
    return X


def image_get_minmax():
    return 0.0, 1.0

def proj_get_minmax():
    return 0.0, 4.0

def normalize(data, minmax):
    data_min, data_max = minmax
    data = np.clip(data, data_min, data_max)
    data = (data - data_min) / (data_max - data_min)
    data = data * 255.0
    data = data.astype(np.float32)
    data = np.expand_dims(np.transpose(np.expand_dims(data, 2), (2, 0, 1)),0)
    return data

def to_numpy(data):
    data = data.detach().cpu().numpy()
    data = data.squeeze()
    if data.ndim == 3: data = data.transpose(1, 2, 0)
    return data

def imwrite(idx, dir, datalist):
    for i in range(len(datalist)):
        file_dir = dir[i] + str(idx)+'.png'
        plt.imsave(file_dir, datalist[i].data.cpu().numpy().squeeze(), cmap="gray")

param = initialization()
ray_trafo, FBPOper, op_norm = build_gemotry(param)
op_modfp = odl_torch.OperatorModule(ray_trafo)
test_mask = np.load(os.path.join(opt.data_path, 'testmask.npy'))


def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])



def test_image(data_path, imag_idx, mask_idx, testproj):
    txtdir = os.path.join(data_path, 'test_640geo_dir.txt')
    mat_files = open(txtdir, 'r').readlines()
    gt_dir = mat_files[imag_idx]
    file_dir = gt_dir[:-6]
    data_file = file_dir + str(mask_idx) + '.h5'
    abs_dir = os.path.join(data_path, 'test_640geo/', data_file)
    gt_absdir = os.path.join(data_path, 'test_640geo/', gt_dir[:-1])
    gt_file = h5py.File(gt_absdir, 'r')
    Xgt = gt_file['image'][()]
    gt_file.close()
    file = h5py.File(abs_dir, 'r')
    Xma= file['ma_CT'][()]
    Sma = file['ma_sinogram'][()]
    XLI = file['LI_CT'][()]
    SLI = file['LI_sinogram'][()]
    TrMAR = file['metal_trace'][()]
    Sgt = np.asarray(ray_trafo(Xgt))
    file.close()

    Np, Nb = Sma.shape  
    D = np.zeros((Np, Nb), dtype=float)  
    factor = Np // testproj  
    D[::factor, :] = 1
    DI = 1 - D  
    Smasp = D * Sma  
    SLIsp = D * SLI
    Xmasp = FBPOper(Smasp)
    XLIsp = FBPOper(SLIsp)
    Sgt = np.asarray(ray_trafo(Xgt))
    M512 = test_mask[:,:,mask_idx]
    M = np.array(Image.fromarray(M512).resize((416, 416), PIL.Image.BILINEAR))
    Xma = normalize(Xmasp, image_get_minmax())  
    Xgt = normalize(Xgt, image_get_minmax())
    XLI = normalize(XLIsp, image_get_minmax())
    Sma = normalize(Smasp, proj_get_minmax())
    Sgt = normalize(Sgt, proj_get_minmax())
    SLI = normalize(SLIsp, proj_get_minmax())

    TrI_bool = np.logical_or(TrMAR,
                             DI)  
    TrI = np.zeros((Np, Nb), dtype=float)
    TrI[TrI_bool == True] = 1
    TrDC = 1 - TrI 
    Tr = TrDC.astype(np.float32)
    Tr = np.expand_dims(np.transpose(np.expand_dims(Tr, 2), (2, 0, 1)),0)
    Mask = M.astype(np.float32)
    Mask = np.expand_dims(np.transpose(np.expand_dims(Mask, 2), (2, 0, 1)),0)
    return torch.Tensor(Xma).cuda(), torch.Tensor(XLI).cuda(), torch.Tensor(Xgt).cuda(), torch.Tensor(Mask).cuda(), \
       torch.Tensor(Sma).cuda(), torch.Tensor(SLI).cuda(), torch.Tensor(Sgt).cuda(), torch.Tensor(Tr).cuda(), torch.Tensor(D).cuda()

def sino_norm(X):
    X = torch.clamp(X,0,1.0)
    return X

def print_network(name, net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    #  print(net)
    print('name={:s}, Total number={:d}'.format(name, num_params))

def main():
    # Build model
    print('Loading model ...\n')
    net = MEPNet(opt).cuda()
    print_network("MEPNet", net)
    net.eval()
    net.load_state_dict(torch.load(os.path.join(opt.model_dir)))
    time_test = 0
    count = 0
    psnr_per_epochadn = 0
    ssim_per_epochadn = 0

    for imag_idx in range(200):
        print(imag_idx)
        for mask_idx in range(10):
            Xma, XLI, Xgt, M, Sma, SLI, Sgt, Tr, D = test_image(opt.data_path, imag_idx, mask_idx, opt.test_proj)
            with torch.no_grad():
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()
                ListX, ListYS= net(Xma, XLI,M,Sma, SLI, Tr)
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time
            Xoutclip = torch.clamp(ListX[-1] / 255.0, 0, 0.5)
            Xgtclip = torch.clamp(Xgt / 255.0, 0, 0.5)
            Xmaclip = torch.clamp(Xma / 255.0, 0, 0.5)
            Xoutnorm = Xoutclip / 0.5
            Xgtnorm = Xgtclip / 0.5
            Xmanorm = Xmaclip /0.5
            idx = imag_idx *10+ mask_idx  + 1
            Xnorm = [Xoutnorm, Xmanorm]
            dir = [out_dir, input_dir]

            imwrite(idx, dir, Xnorm)
            psnr_iteradn = psnr(to_numpy(Xoutnorm * (1 - M)), to_numpy(Xgtnorm * (1 - M)), data_range=1)
            psnr_per_epochadn += psnr_iteradn
            ssim_iteradn = ssim(to_numpy(Xoutnorm * (1 - M)), to_numpy(Xgtnorm* (1 - M)), data_range=1)
            ssim_per_epochadn += ssim_iteradn

            end_time = time.time()
            print('Times: ', dur_time)
            count += 1
    print('Avg. time={:.4f}, Avg. OnlyInferencentime={:.4f},  Avg.PSNRadn={:.4f}, Avg.SSIMadn={:.5f}'.format(time_test/count, time_test/count, psnr_per_epochadn/count, ssim_per_epochadn/count))
    print(100*'*')
if __name__ == "__main__":
    main()

