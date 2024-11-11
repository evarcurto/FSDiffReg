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

# Assuming Metrics and diffusion are already defined and properly set up
# And test_loader is available

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

    #registDice = np.zeros((len(test_set), 5))
    #originDice = np.zeros((len(test_set), 5))
    registTime = []
    print('Begin Model Evaluation.')
    idx = 0
    result_path = '{}'.format(opt['path']['results'])

os.makedirs(result_path, exist_ok=True)
print(len(test_loader))

for istep, test_data in enumerate(test_loader):
    dataName = istep
    time1 = time.time()
    
    diffusion.feed_data(test_data)
    diffusion.test_registration()
    time2 = time.time()
    
    visuals = diffusion.get_current_registration()
    
    # Adjusting for 2D images
    defm_frames_visual = visuals['contD'].numpy().transpose(1, 2, 0)  # [height, width, channels]
    flow_frames = visuals['contF'].numpy().transpose(1, 2, 0)  # [height, width, channels]
    flow_frames_ES = flow_frames[:, :, -1]  # Selecting the last channel if needed
    
    sflow = torch.from_numpy(flow_frames_ES).permute(2, 0, 1)  # [channels, height, width]
    sflow = Metrics.transform_grid(sflow[0], sflow[1])  # Adapted for 2D
    
    nh, nw = sflow.shape[1], sflow.shape[2]
    segflow = torch.FloatTensor(sflow.shape).zero_()
    segflow[1] = (sflow[0] / (nh - 1) - 0.5) * 2.0  # H
    segflow[0] = (sflow[1] / (nw - 1) - 0.5) * 2.0  # W
    
    origin_seg = test_data['MS']
    origin_seg = origin_seg.unsqueeze(0)  # Add batch dimension
    
    regist_seg = F.grid_sample(origin_seg.cuda().float(), segflow.cuda().float().permute(1, 2, 0).unsqueeze(0), mode='nearest')
    regist_seg = regist_seg.squeeze().cpu().numpy().transpose(1, 2, 0)
    
    label_seg = test_data['FS'].cpu().numpy()
    origin_seg = test_data['MS'].cpu().numpy()
    
    vals_regist = Metrics.dice_ACDC(regist_seg, label_seg)[::3]
    vals_origin = Metrics.dice_ACDC(origin_seg, label_seg)[::3]
    
    registDice[istep] = vals_regist
    originDice[istep] = vals_origin
    
    print('---- Original Dice: %03f | Deformed Dice: %03f' % (np.mean(vals_origin), np.mean(vals_regist)))
    
    registTime.append(time2 - time1)
    time.sleep(1)
    
omdice, osdice = np.mean(originDice), np.std(originDice)
mdice, sdice = np.mean(registDice), np.std(registDice)
mtime, stime = np.mean(registTime), np.std(registTime)
