from torch import nn
import torch
import torchvision
from skimage import io
import os
import matplotlib.pyplot as plt
from torchvision.io import read_image
import torchvision.transforms as T
import data.util_2D as Util
import numpy as np


#---------------------------------IMAGEM DE ENTRADA--------------------------------------------

# Imagem criada através de geração uma matriz 4x4 com cores aleatórias em RGB
#imagem_rgb_4x4 = np.random.rand(4, 4, 3)
#print(imagem_rgb_4x4)

# Criar a imagem com as cores RGB geradas
#plt.imshow(imagem_rgb_4x4)
#plt.axis('off')  # Retirar os eixos
#plt.show()
#exit()

image_4x4 = np.array([[[0.28033062, 0.63666627, 0.50278588],
                   [0.27536151, 0.14543703, 0.38348563],
                   [0.0238481,  0.2172431,  0.1796081 ],
                   [0.3105019,  0.1427701,  0.69136378]],

                  [[0.4597945,  0.04457584, 0.39559152],
                   [0.09126704, 0.95714992, 0.38508917],
                   [0.3890701,  0.30023762, 0.3490191 ],
                   [0.9333373,  0.60074979, 0.8501193 ]],

                  [[0.29380664, 0.59143301, 0.34433129],
                   [0.89060807, 0.01443899, 0.00755597],
                   [0.66777375, 0.08236226, 0.85818518],
                   [0.8948939,  0.62506005, 0.64544333]],

                  [[0.38616123, 0.88149673, 0.65841355],
                   [0.73065238, 0.62745326, 0.84589424],
                   [0.27643877, 0.09363761, 0.13527019],
                   [0.51623343, 0.19737761, 0.20088305]]])


#np.set_printoptions(precision=3)
#print(input)
#plt.imshow(input)
#plt.axis('off')  # Retirar os eixos
#plt.show()
#exit()

#Caso queira ler outra imagem de entrada
dataXPath = './DEBUG/input.png'
data = io.imread(dataXPath).astype(float)/255

#---------------------------------RESHAPES da IMAGEM/TENSOR--------------------------------------------

#Transformar imagem num tensor
#[data, label] = Util.transform_augment([data, data], split='train', min_max=(-1, 1))
#print(image_4x4.shape)
#input_ = image_4x4.transpose(2,0,1)
input_ = data.transpose(2,0,1)
input = torch.from_numpy(input_).contiguous()
#print('input transposed', input)
#print(input.shape)

torchvision.utils.save_image(input, 'DEBUG/input_data_tensor.png', normalize = True)

img = input.unsqueeze(0) #Adicionar dimensão de batchsize

nbatch = img.shape[0]
nch    = img.shape[1]
height = img.shape[2]
width  = img.shape[3]

#print(img)
#print(img.shape)

#Achatar tensor numa única linha --> [b,c,h,w]->[b*c*h*w,1]
#img_flat = img.permute(0,2,3,1)
#img_flat = img.view(-1,3).float().cuda()
#img_flat = img_flat.permute(1,0)
#print(img_flat)
#exit()
#plt.imshow(img_flat,cmap='gray')
#plt.colorbar() 
#plt.axis('off')  # Retirar os eixos
#plt.show()
#exit()
#print(img_flat.shape)
#print(img_flat)
#exit()

#Transformar de novo o tensor na sua forma anterior -> [b,c,h,w]
#output=img_flat
#O = output.view(nbatch,nch, height, width)
#print(O)
#torchvision.utils.save_image(O, 'DEBUG/output_image_4x4.png', normalize = True)
#O = output.view(nbatch, height, width, nch).permute(0, 3, 1, 2)  #B, C, H, W  #linha de código do código original que eu acho que não está correcta

#Transformar tensor output num narray para efeitos de visualização
#output = O.squeeze()
#output = output.permute(1,2,0)
#plt.imshow(output)
#plt.axis('off')  # Retirar os eixos
#plt.show()

#exit()
#---------------------------------SELECÇAO INDICES--------------------------------------------

#Gerar meshgrid
hgt = height
wdt = width
h_t = torch.matmul(torch.linspace(0.0, hgt-1.0, hgt).unsqueeze_(1), torch.ones((1,wdt))).cuda()
w_t = torch.matmul(torch.ones((hgt,1)), torch.linspace(0.0, wdt-1.0, wdt).unsqueeze_(1).transpose(1,0)).cuda()

H_mesh = h_t.unsqueeze_(0).expand(nbatch, hgt, wdt)
W_mesh = w_t.unsqueeze_(0).expand(nbatch, hgt, wdt)

#flow_H = torch.rand(1,hgt,wdt)
#flow_H = flow_H.cuda()
# flow_H = torch.tensor([[[0.0069, 0.0329, 0.0570, 0.0575],
#          [0.0294, 0.0864, 0.0102, 0.0910],
#          [0.0901, 0.0565, 0.0327, 0.0270],
#          [0.0130, 0.0099, 0.0098, 0.0348]]]).cuda()

flow_H = (torch.ones(1,hgt,wdt)*20).cuda()
flow_W = (torch.ones(1,hgt,wdt)*20).cuda()
#flow_W = torch.rand(1,hgt,wdt)
#flow_W = flow_W.cuda()
#print(flow_H)
#print(flow_W)

H_upmesh = flow_H + H_mesh
W_upmesh = flow_W + W_mesh

input = img
nbatch = input.shape[0]
nch    = input.shape[1]
height = input.shape[2]
width  = input.shape[3]

img = torch.zeros(nbatch, nch, height+2, width+2).cuda()  #[1,3,6,6]
img[:, :, 1:-1, 1:-1] = input
img[:, :, 0, 1:-1] = input[:, :, 0, :]
img[:, :, -1, 1:-1] = input[:, :, -1, :]
img[:, :, 1:-1, 0] = input[:, :, :, 0]
img[:, :, 1:-1, -1] = input[:, :, :, -1]
img[:, :, 0, 0] = input[:, :, 0, 0]
img[:, :, 0, -1] = input[:, :, 0, -1]
img[:, :, -1, 0] = input[:, :, -1, 0]
img[:, :, -1, -1] = input[:, :,-1, -1]

img_flat = img.permute(0,2,3,1)
img_flat = img_flat.view(-1,3).float()

#torchvision.utils.save_image(img, 'DEBUG/indices_test/img_6x6_tensor.png', normalize = True)
#Transformar tensor output num narray para efeitos de visualização
#output = img.squeeze()
#output = output.permute(1,2,0).cpu()
#plt.imshow(output)
#plt.axis('off')  # Retirar os eixos
#plt.show()

imgHgt = img.shape[2] #6
imgWdt = img.shape[3] #6

# H_upmesh, W_upmesh = [H, W] -> [BHW,] 
H_upmesh = H_upmesh.view(-1).float()+1.0  # (BHW,)
W_upmesh = W_upmesh.view(-1).float()+1.0  # (BHW,)

#Identification of the four neighboring pixels around a target point.
#The target point is located at non-integer coordinates, and the goal of bilinear interpolation is to compute a value at this point by considering its neighboring pixels.
#These neighbors are the pixels at integer coordinates around the target pixel.
hf = torch.floor(H_upmesh).int()
hc = hf + 1
wf = torch.floor(W_upmesh).int()
wc = wf + 1

# H_upmesh, W_upmesh -> Clamping
hf = torch.clamp(hf, 0, imgHgt-1)  # (BHW,)
hc = torch.clamp(hc, 0, imgHgt-1)  # (BHW,)
wf = torch.clamp(wf, 0, imgWdt-1)  # (BHW,)
wc = torch.clamp(wc, 0, imgWdt-1)  # (BHW,)

# Find batch indexes
#o tensor rep serve para replicar os indices do batch para todos os pixeis da imagem
rep = torch.ones([height*width, ]).unsqueeze_(1).transpose(1, 0).cuda()
bHW = torch.matmul((torch.arange(0, nbatch).float()*imgHgt*imgWdt).unsqueeze_(1).cuda(), rep).view(-1).int()

# Box updated indexes
W = imgWdt
# x: W, y: H, z: D
idx_00 = bHW + hf*W + wf
idx_10 = bHW + hf*W + wc
idx_01 = bHW + hc*W + wf
idx_11 = bHW + hc*W + wc


val_00 = torch.index_select(img_flat, 0, idx_00.long())
val_10 = torch.index_select(img_flat, 0, idx_10.long())
val_01 = torch.index_select(img_flat, 0, idx_01.long())
val_11 = torch.index_select(img_flat, 0, idx_11.long())


dHeight = hc.float() - H_upmesh
dWidth  = wc.float() - W_upmesh

wgt_00 = (dHeight*dWidth).unsqueeze_(1)
wgt_10 = (dHeight * (1-dWidth)).unsqueeze_(1)
wgt_01 = ((1-dHeight) * dWidth).unsqueeze_(1)
wgt_11 = ((1-dWidth) * (1-dHeight)).unsqueeze_(1)
output = val_00*wgt_00 + val_10*wgt_10 + val_01*wgt_01 + val_11*wgt_11

output = output.view(nbatch, height, width, nch)
output = output.permute(0,3,1,2)
#output = output.view(nbatch, nch, height, width)
torchvision.utils.save_image(output, 'DEBUG/output_data_tensor_trans20.png', normalize = True)

#Transformar tensor output num narray para efeitos de visualização
#output = img.squeeze()
#output = output.permute(1,2,0).cpu()
#plt.imshow(output)
#plt.axis('off')  # Retirar os eixos
#plt.show()