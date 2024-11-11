import wandb
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import torch
import data as Data
import model as Model
import argparse
import logging
import core.logger as Logger
import os
from math import *
import time

from util.visualizer import Visualizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/train_2D.json',
                        help='JSON file for configuration')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)

    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    # Convert to NoneDict, which return None for missing key.
    opt = Logger.dict_to_nonedict(opt)
    visualizer = Visualizer(opt)
    # Initialize W&B project
    #wandb.init(project="my_project_name", notes="1channels", settings=wandb.Settings(code_dir="."), config=opt)
    wandb.init(project="my_project_name", notes="1channels", config=opt)
    wandb.run.log_code(".")


    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # dataset
    phase = 'train'
    dataset_opt = opt['datasets']['train']
    batchSize = opt['datasets']['train']['batch_size']
    train_set = Data.create_dataset_2D_MyDataset(dataset_opt, phase)
    train_loader = Data.create_dataloader(train_set, dataset_opt, phase)
    training_iters = int(ceil(train_set.data_len / float(batchSize)))
    print('Dataset Initialized')

    # model
    diffusion = Model.create_model(opt)
    print("Model Initialized")

    # Train
    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_epoch = opt['train']['n_epoch']
    if opt['path']['resume_state']:
        print('Resuming training from epoch: {}, iter: {}.'.format(current_epoch, current_step))

    while current_epoch < n_epoch:
        current_epoch += 1
        for istep, train_data in enumerate(train_loader):
            iter_start_time = time.time()
            current_step += 1
            diffusion.feed_data(train_data)
            diffusion.optimize_parameters()
            t = (time.time() - iter_start_time) / batchSize
            
            # Log message
            message = f'(epoch: {current_epoch} | iters: {istep+1}/{training_iters} | time: {t:.3f}) '
            errors = diffusion.get_current_log()
            for k, v in errors.items():
                message += f'{k}: {v:.6f} '
            print(message)

            # Log metrics to W&B
            wandb.log(errors)
            wandb.log({"epoch": current_epoch, "step_time": t})

            if (istep + 1) % opt['train']['print_freq'] == 0:
                logs = diffusion.get_current_log()
                visualizer.print_current_errors(current_epoch, istep+1, training_iters, logs, t, 'Train')
                visualizer.plot_current_errors(current_epoch, (istep+1) / float(training_iters), logs)
                visuals = diffusion.get_current_visuals_train_bt()
                visualizer.display_current_results(visuals, current_epoch, istep, True)

        # Save model checkpoints to W&B
        if current_epoch % opt['train']['save_checkpoint_epoch'] == 0:
            diffusion.save_network(current_epoch, current_step)
            wandb.save(f'model_epoch_{current_epoch}.pth')  # Saves the model to W&B

