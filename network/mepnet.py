# -*- coding: utf-8 -*-
"""
Created on July 5 22:40:20 2022

@author: hazelhwang
"""

"""
symbol explanations: 
code    paper        meanings
primal    X          CT image
dual      \bar{S}    normalized sinogram data
Ys        \bar{Y}    normalized coefficient
Ysf       S          sinogram data
"""


# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as  F
from torch.autograd import Variable
import scipy.io as io
import odl
from odl.contrib import torch as odl_torch
import random
from network.priornet import UNet
from utils.build_gemotry import initialization, build_gemotry
from network.F_Conv import *

para_ini = initialization()
fp, fbp, opnorm = build_gemotry(para_ini)
op_modfp = odl_torch.OperatorModule(fp)
op_modfbp = odl_torch.OperatorModule(fbp)
op_modpT = odl_torch.OperatorModule(fp.adjoint)

filter = torch.FloatTensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]) / 9
filter = filter.unsqueeze(dim=0).unsqueeze(dim=0)

class MEPNet(nn.Module):
    def __init__(self, args):
        super(MEPNet, self).__init__()
        self.S = args.S  # Stage number S includes the initialization process
        self.iter = self.S - 1  # not include the initialization process
        self.num_u = args.num_channel + 1  # concat extra 1 term
        self.num_f = args.num_channel + 2  # concat extra 2 terms
        self.eta1const = args.eta1
        self.eta2const = args.eta2
        self.alphaconst = args.alpha

        # Stepsize
        self.eta1 = torch.Tensor([self.eta1const])                                    
        self.eta2 = torch.Tensor([self.eta2const])                                   
        self.alpha = torch.Tensor([self.alphaconst])
        self.eta1S = self.make_coeff(self.S, self.eta1)                           
        self.eta2S = self.make_coeff(self.S, self.eta2)
        self.alphaS = self.make_coeff(self.S, self.alpha)

        self.T = args.T

        # proxNet
        self.priornet = UNet(n_channels=2, n_classes=1, n_filter=32)
        self.proxNet_u_S = self.make_primalnet(self.S, args.num_channel+1, self.T)
        self.proxNet_f_S = self.make_dualnet(self.S, args.num_channel+1, self.T)
        self.proxNet_u0 = CTnet(args.num_channel+1, self.T)  # fine-tune at the last
        self.proxNet_f0 = Projnet(args.num_channel+1, self.T)  # fine-tune at the last


        # filter for initializing B and Z
        self.Cu_const = filter.expand(args.num_channel, 1, -1, -1)  # size: 1*1*3*3
        self.Cu = nn.Parameter(self.Cu_const, requires_grad=True)
        self.Cf_const = filter.expand(args.num_channel, 1, -1, -1)  # size:1*1*3*3
        self.Cf = nn.Parameter(self.Cf_const, requires_grad=True)
        self.bn = nn.BatchNorm2d(1)
    def make_coeff(self, iters,const):
        const_dimadd = const.unsqueeze(dim=0)
        const_f = const_dimadd.expand(iters,-1)
        coeff = nn.Parameter(data=const_f, requires_grad = True)
        return coeff


    def make_primalnet(self, iters, channel, T):  #
        layers = []
        for i in range(iters):
            layers.append(CTnet(channel, T))
        return nn.Sequential(*layers)

    def make_dualnet(self, iters, channel, T):
        layers = []
        for i in range(iters):
            layers.append(Projnet(channel, T))
        return nn.Sequential(*layers)

    def make_conv(self, times, channel):
        layers = []
        for i in range(times):
            layers.append(nn.Sequential(
                nn.Conv2d(1, channel, kernel_size=3, stride=1, padding=1, dilation=1)
            ))
        return nn.Sequential(*layers)

    def make_convop(self, times, channel):
        layers = []
        for i in range(times):
            layers.append(nn.Sequential(
                nn.Conv2d(channel, channel, kernel_size=3, stride=1, padding=1, dilation=1)
            ))
        return nn.Sequential(*layers)

    def forward(self, Xma, XLI, M, Sma, SLI, Tr):
        # save mid-updating results
        Listu = []
        Listysf = []
        # initialize primal and dual
        UZ00 = F.conv2d(XLI,  self.Cu, stride=1, padding=1)  # dual variable z
        input_ini = torch.cat((XLI, UZ00), dim=1)
        UZ_ini = self.proxNet_u0(input_ini)
        U0 = UZ_ini[:, :1, :, :]
        UZ = UZ_ini[:, 1:, :, :]
        primal = U0


        FZ00 = F.conv2d(SLI, self.Cf, stride=1, padding=1)  # dual variable z
        input_ini = torch.cat((SLI, FZ00), dim=1)
        FZ_ini= self.proxNet_f0(input_ini)
        F0 = FZ_ini[:, :1, :, :]
        FZ = FZ_ini[:, 1:, :, :]
        dual = F0


        # PriorNet
        prior_input = torch.cat((Xma, XLI), dim=1)
        Us = XLI + self.priornet(prior_input)
        Ys = op_modfp(F.relu(self.bn(Us)) / 255)
        Us = F.relu(self.bn(Us))
        Ys = Ys / 4.0 * 255
        PU = op_modfp(primal/255)/ 4.0 * 255
        ESf =  Ys*dual - PU
        Gf = Ys*ESf + self.alphaS[0]*Tr * Tr * Ys * (Ys * dual - Sma)

        # print("Gf,", Gf.shape)
        f_dual = dual - self.eta1S[0]/10*Gf
        input_fdual = torch.cat((f_dual, FZ), dim=1)
        out_fdual= self.proxNet_f_S[0](input_fdual)
        dual = out_fdual[:,:1,:,:]
        FZ =  out_fdual[:,1:,:,:]
        Listysf.append(Ys * dual)

        #1st iteration: Updating f1--U1
        ESu = PU - Ys * dual 
        Gu = op_modpT((ESu/255) * 4.0)
        u_dual = primal - self.eta2S[0] / 10 * Gu
        input_udual = torch.cat((u_dual, UZ), dim=1)
        out_udual = self.proxNet_u_S[0](input_udual)
        primal = out_udual[:, :1, :, :]
        UZ = out_udual[:, 1:, :, :]
        Listu.append(primal)
        for i in range(self.iter):
            # 1st iteration: Updating U-->f
            PU = op_modfp(primal / 255) / 4.0 * 255
            ESf = Ys * dual - PU
            Gf = Ys * ESf  + self.alphaS[i+1] * Tr * Tr * Ys * (Ys * dual - Sma)
            f_dual = dual - self.eta1S[i+1] / 10 * Gf
            input_fdual = torch.cat((f_dual, FZ), dim=1)
            out_fdual= self.proxNet_f_S[i+1](input_fdual)
            dual = out_fdual[:, :1, :, :]
            FZ = out_fdual[:, 1:, :, :]
            Listysf.append(Ys * dual)
            # 1st iteration: Updating f--U
            ESu = PU - Ys *dual
            Gu = op_modpT((ESu / 255) * 4.0)
            #Gu = op_modpT(ESu)
            u_dual = primal - self.eta2S[i+1] / 10 * Gu
            input_udual = torch.cat((u_dual, UZ), dim=1)
            out_udual = self.proxNet_u_S[i+1](input_udual)
            primal = out_udual[:, :1, :, :]
            UZ = out_udual[:, 1:, :, :]
            Listu.append(primal)
        return Listu, Listysf

# proxNet_f: sinogram domain
class Projnet(nn.Module):
    def __init__(self, channel, T):
        super(Projnet, self).__init__()
        self.channels = channel
        self.T = T  
        self.layer = self.make_resblock(self.T) 
    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(
                nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              nn.ReLU(),
                              nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                              nn.BatchNorm2d(self.channels),
                              ))
        return nn.Sequential(*layers)

    def forward(self, input):
        dual = input
        for i in range(self.T):
            dual = F.relu(dual + self.layer[i](dual))
        return dual



#proxNet_u
class CTnet(nn.Module):
    def __init__(self, channel, T):
        super(CTnet, self).__init__()
        self.channels = channel  # 3 means R,G,B channels for color image
        self.T = T
        self.layer = self.make_resblock(self.T)
      
    def make_resblock(self, T):
        layers = []
        for i in range(T):
            layers.append(nn.Sequential(
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
                nn.ReLU(),
                nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
                nn.BatchNorm2d(self.channels),
            ))
        return nn.Sequential(*layers)

    def forward(self, input):
        primal = input
        for i in range(self.T):
            primal = F.relu(primal + self.layer[i](primal))
        return primal



# # proxNet_X: CNN in InDuDoNet
# class CTnet(nn.Module):
#     def __init__(self, channel, T):
#         super(CTnet, self).__init__()
#         self.channels = channel
#         self.T = T
#         self.layer = self.make_resblock(self.T)
#     def make_resblock(self, T):
#         layers = []
#         for i in range(T):
#             layers.append(nn.Sequential(
#                 nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#                 nn.BatchNorm2d(self.channels),
#                 nn.ReLU(),
#                 nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1),
#                 nn.BatchNorm2d(self.channels),
#             ))
#         return nn.Sequential(*layers)

#     def forward(self, input):
#         X = input
#         for i in range(self.T):
#             X = F.relu(X + self.layer[i](X))
#         return X



class CTnet(nn.Module):
    def __init__(self, channel, T):
        super(CTnet, self).__init__()
        self.channels = channel
        n_feats = 4
        tranNum = 8  
        self.head_in = nn.Sequential(Fconv_PCA(5, self.channels, n_feats, tranNum, inP=5, padding=(5-1)//2, ifIni=1, Smooth=False, iniScale=1.0))
        module_body = [ResBlock(Fconv_PCA, n_feats, 5, tranNum=tranNum, inP=5, bn=True, act=nn.ReLU(True), res_scale=0.1, Smooth=False, iniScale=1.0
            ) for _ in range(T-1)
        ]
        self.body = nn.Sequential(*module_body)
        self.out = nn.Sequential(Fconv_PCA_out(5, n_feats, self.channels, tranNum, inP=5, padding=(5-1)//2, ifIni=0, Smooth=False, iniScale=1.0))
        self.adjust_layer = nn.Sequential(nn.Conv2d(self.channels, self.channels, kernel_size=3, stride=1, padding=1, dilation=1))
    def forward(self, input):
        x = self.head_in(input)
        res = self.body(x)
        res = self.out(res)
        x = F.relu(input + 0.01 * self.adjust_layer(res))
        return x







