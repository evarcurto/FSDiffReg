import os
#os.environ["CUDA_VISIBLE_DEVICES"]="3"
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import core.metrics as Metrics
from math import *
import time
import numpy as np
import torch.nn.functional as F
from PIL import Image

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy.astype('uint8'))
    image_pil.save(image_path)

from util import util

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/.json',
                        help='JSON file for configuration')
    #parser.add_argument('-w', '--weights', type=str, default='',
    #                    help='weights file for validation')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    phase = 'test'
    dataset_opt=opt['datasets']['test']
    test_set = Data.create_dataset_2D_test(dataset_opt, phase)
    test_loader = Data.create_dataloader(test_set, dataset_opt, phase)
    print('Dataset Initialized')

    #opt['path']['resume_state']=args.weights
    # model
    diffusion = Model.create_model(opt)
    print("Model Initialized")
    # Train

    registTime = []
    print('Begin Model Evaluation.')
    idx = 0
    result_path = '{}'.format(opt['path']['results'])

# Função para salvar os valores de cada canal de uma imagem RGB em ficheiros de texto separados
def save_image_channels_to_txt(tensor, base_filepath):
    # Verifica se o tensor tem o formato esperado [1, 3, height, width]
    if len(tensor.shape) != 4 or tensor.shape[1] != 3 or tensor.shape[0] != 1:
        raise ValueError("O tensor deve ter o formato [1, 3, height, width].")

    _, channels, height, width = tensor.shape
    
    for channel_index in range(channels):
        filepath = f"{base_filepath}_channel_{channel_index+1}.txt"
        # Abrir o ficheiro de texto para escrita
        with open(filepath, 'w') as f:
            for h in range(height):
                for w in range(width):
                    pixel_value = tensor[0, channel_index, h, w].item()
                    f.write(f"{pixel_value:.2f} ")
                f.write("\n")
        
    os.makedirs(result_path, exist_ok=True)
    #print(len(test_loader))
for istep,  test_data in enumerate(test_loader):
            idx += 1
            fileInfo = test_data['P']
            dataXinfo, dataYinfo = fileInfo[0][0][:-4], fileInfo[1][0][:-4]

            time1 = time.time()
            diffusion.feed_data(test_data)

            print('Generation from %s to %s' % (dataXinfo, dataYinfo))
            # diffusion.test_generation(continuous=True)
            print('Registration from %s to %s' % (dataXinfo, dataYinfo))
            diffusion.test_registration(continuous=True)
            time2 = time.time()

            # Code to save moving and fixed images
            data_origin = test_data['M'].squeeze().cpu().numpy()
            data_fixed = test_data['F'].squeeze().cpu().numpy()
            data_origin = (data_origin+1)/2. * 255
            data_fixed = (data_fixed + 1) / 2. * 255
            savePath = os.path.join(result_path, '%s_TO_%s_mov.png' % (dataXinfo, dataYinfo))
            # save_image(data_origin, savePath)
            save_image(data_origin.transpose(1,2,0), savePath)
            savePath = os.path.join(result_path, '%s_TO_%s_fix.png' % (dataXinfo, dataYinfo))
            # save_image(data_fixed, savePath)
            save_image(data_fixed.transpose(1,2,0), savePath)

            #-------------Registration results 
            visuals = diffusion.get_current_registration()
            #print(visuals['contD'].shape)
            #print(visuals['contF'].shape)
            defm_frames_visual = visuals['out_M']
            flow_frames = visuals['flow']
            # Caminho para o ficheiro de texto
            output_filepath = 'image_tensor_values'
            save_image_channels_to_txt(flow_frames, output_filepath)
            savePath=os.path.join(result_path, 'out.png')
            util.save_image_torch(defm_frames_visual, savePath)
            savePath=os.path.join(result_path, 'flow.png')
            util.save_image_torch(flow_frames, savePath)
            exit()

            # #-------------Generation results
            # visuals = diffusion.get_current_generation()
            # sample_data = visuals['MF'].squeeze().numpy()
            # #print("----")
            # #print(sample_data.shape[0])
            # for isamp in range(0, sample_data.shape[0], 2):
            #     savePath = os.path.join(result_path, '%s_TO_%s_sample_%d.png' % (dataXinfo, dataYinfo, isamp))
            #     synthetic_data = sample_data[isamp]
            #     synthetic_data -= synthetic_data.min()
            #     synthetic_data /= synthetic_data.max()
            #     synthetic_data = synthetic_data * 255
            #     save_image(synthetic_data, savePath)
            # savePath = os.path.join(result_path, '%s_TO_%s_sample_last.png' % (dataXinfo, dataYinfo))
            # synthetic_data = sample_data[-1]
            # synthetic_data -= synthetic_data.min()
            # synthetic_data /= synthetic_data.max()
            # synthetic_data = synthetic_data * 255
            # save_image(synthetic_data, savePath)

  
     


  