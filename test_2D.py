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
        
    os.makedirs(result_path, exist_ok=True)
    print(len(test_loader))
for istep,  test_data in enumerate(test_loader):
            idx += 1
            fileInfo = test_data['P']
            dataXinfo, dataYinfo = fileInfo[0][0][:-4], fileInfo[1][0][:-4]

            data_origin = test_data['M'].squeeze().cpu().numpy()
            data_fixed = test_data['F'].squeeze().cpu().numpy()
            time1 = time.time()
            diffusion.feed_data(test_data)
            print('Registration from %s to %s' % (dataXinfo, dataYinfo))
            diffusion.test_registration()
            time2 = time.time()
            visuals = diffusion.get_current_registration()

            #data_origin = (data_origin+1)/2. * 255
            #data_fixed = (data_fixed + 1) / 2. * 255
            savePath = os.path.join(result_path, '%s_TO_%s_mov.png' % (dataXinfo, dataYinfo))
            save_image(data_origin, savePath)
            savePath = os.path.join(result_path, '%s_TO_%s_fix.png' % (dataXinfo, dataYinfo))
            save_image(data_fixed, savePath)


            #print(visuals['contD'].shape)
            #print(visuals['contF'].shape)
            defm_frames_visual = visuals['contD']
            flow_frames = visuals['contF']
       
            savePath=os.path.join(result_path, 'out.png')
            util.save_image_torch(defm_frames_visual, savePath)
            savePath=os.path.join(result_path, 'flow.png')
            util.save_image_torch(flow_frames, savePath)


  
     


  