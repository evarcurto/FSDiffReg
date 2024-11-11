import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time

class gradientLoss(nn.Module):
    def __init__(self, penalty='l1'):
        super(gradientLoss, self).__init__()
        self.penalty = penalty

    def forward(self, input):

        dH = torch.abs(input[:, :, 1:, :] - input[:, :, :-1, :])
        dW = torch.abs(input[:, :, :, 1:] - input[:, :, :, :-1])
        if(self.penalty == "l2"):

            dH = dH * dH
            dW = dW * dW
        loss = (torch.mean(dH) + torch.mean(dW)) / 2.0
        return loss
    
class crossCorrelation2D(nn.Module):
    def __init__(self, in_ch, kernel=(9, 9), gamma=1):
        super(crossCorrelation2D, self).__init__()
        self.in_ch = in_ch
        self.kernel = kernel
        self.gamma=gamma
        self.filt = (torch.ones([1, in_ch, self.kernel[0], self.kernel[1]])).to('cuda:0')
        #self.filt = torch.nn.Conv2d(3, 3, kernel_size=(9, 9))
        #self.filt = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(9, 9), bias=False).to('cuda:0')



    def forward(self, input, target,flow):
        min_max = (-1, 1)
        target = (target - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
        
        II = input * input
        TT = target * target
        IT = input * target

        flow=F.sigmoid(flow)**self.gamma
        pad = (int((self.kernel[0]-1)/2), int((self.kernel[1]-1)/2))
        T_sum = F.conv2d(target, self.filt, stride=1, padding=pad)
        I_sum = F.conv2d(input, self.filt, stride=1, padding=pad)#*flow
        TT_sum = F.conv2d(TT, self.filt, stride=1, padding=pad)
        II_sum = F.conv2d(II, self.filt, stride=1, padding=pad)#*flow
        IT_sum = F.conv2d(IT, self.filt, stride=1, padding=pad)
        kernelSize = self.kernel[0] * self.kernel[1]
        Ihat = I_sum / kernelSize
        That = T_sum / kernelSize

        cross = IT_sum - Ihat*T_sum - That*I_sum + That*Ihat*kernelSize
        T_var = TT_sum - 2*That*T_sum + That*That*kernelSize
        I_var = II_sum - 2*Ihat*I_sum + Ihat*Ihat*kernelSize
        cc = cross*cross*flow / (T_var*I_var+1e-5)

        loss = -1.0 * torch.mean(cc)
        return loss
    