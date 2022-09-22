import os
import numpy as np
from data.coco import CountingkDataset
from torch.utils.data import DataLoader
import torch.nn as nn
from model.CFOCNet import CFOCNet
from torch import optim
from model.loss import my_loss
import torch
import logging
import matplotlib.pyplot as plt
from torchvision.utils import save_image, make_grid

class Runner:
    def __init__(self,args,config):
        self.args= args
        self.config= config
        self.args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(self.args.log_sample_path, exist_ok=True)

    def train(self):

        # Import the dataset
        dataset = CountingkDataset(self.config,'train')
        data_loader = DataLoader(dataset,self.config.train.batch_size,shuffle=True,pin_memory=True,num_workers=self.config.train.num_workers)
        print("dataset length ", len(dataset))

        net = CFOCNet()

        net.to(self.config.device)

        optimizer = optim.Adam(net.parameters(),lr=self.config.optimizer.lr)

        plt.gca().invert_yaxis() # because the direction of y axis in heatmap is different from images


        for epoch in range(self.config.train.epochs):
            for i, sample in enumerate(data_loader):
                optimizer.zero_grad()
                
                queries = sample[0].to(self.config.device)
                references = sample[1].to(self.config.device)
                target = sample[2].to(self.config.device)

                net.train()
                FS = net(queries,references)
                loss, ssim_loss = my_loss(FS,target,self.config)
                
                loss.backward()
                optimizer.step()
                
                print(f"Loss: {loss / self.config.train.batch_size}, SSIM Loss: {ssim_loss / self.config.train.batch_size}")

#                 if i % 1000 == 0:
#                     net.eval()
#                     with torch.no_grad():
#                         predict = net(queries,references)

#                         # Save query image
#                         save_image(queries[0], os.path.join(self.config.train.result_path, f"Epoch_{epoch}_number_{i*self.config.train.batch_size}_query.png")) 

#                         # Save reference images
#                         save_image(references[0], os.path.join(self.config.train.result_path, f"Epoch_{epoch}_number_{i*self.config.train.batch_size}_reference.png")) 

#                         # Save target images
#                         plt.imshow(target[0,0].cpu().numpy())
#                         plt.title(f'target sum : {np.sum(target[0,0].cpu().numpy())}')
#                         plt.savefig(os.path.join(self.config.train.result_path, f"Epoch_{epoch}_number_{i*self.config.train.batch_size}_target.png"))

#                         # Save predict images
#                         plt.imshow(predict[0,0].cpu().numpy())
#                         plt.title(f'predict sum : {np.sum(predict[0,0].cpu().numpy())}')
#                         plt.savefig(os.path.join(self.config.train.result_path, f"Epoch_{epoch}_number_{i*self.config.train.batch_size}_predict.png"))

#                         logging.info(f'Epoch {epoch} Number {i*self.config.train.batch_size} picture loss:{loss.item()/self.config.train.batch_size}')
#                         logging.info(f'target sum: {np.sum(target[0,0].cpu().numpy())}, predict sum: {np.sum(predict[0,0].cpu().numpy())}')


            torch.save(net.state_dict(),os.path.join(self.args.log_path,f'model_epoch_{epoch}.pth'))

    def test(self):
        
        net = CFOCNet()
        
        net.to(self.config.device)

        checkpoint = torch.load(self.config.eval.checkpoint)
        net.load_state_dict(checkpoint)
        net.eval()

       
        dataset = CountingkDataset(self.config,'train',[3])
        data_loader = DataLoader(dataset,self.config.train.batch_size,pin_memory=True,num_workers=self.config.train.num_workers)

        mae_sum = 0
        mse_sum = 0
        
        count = len(dataset)

        with torch.no_grad():
            for i, sample in enumerate(data_loader):

                queries = sample[0].to(self.config.device)
                references = sample[1].to(self.config.device)
                target = sample[2].to(self.config.device)

                predict = net(queries,references)

                logging.info(f'target num: {torch.sum(target).item()}, predict num: {torch.sum(predict).item()}')

                target_num = torch.sum(target).item()
                predict_num = torch.sum(predict).item()

                mae_sum += abs(target_num-predict_num)
                mse_sum += abs(target_num-predict_num) **2

#                 if self.config.eval.sample:

#                     os.makedirs(os.path.join(self.config.eval.image_folder,'bad'),exist_ok=True)
#                     os.makedirs(os.path.join(self.config.eval.image_folder,'good'),exist_ok=True)

#                     if abs(target_num - predict_num) >=50:

#                         save_image(queries[0], os.path.join(self.config.eval.image_folder, f'bad/{i}_query_bad.png'))
#                         # grids = make_grid(references)
#                         save_image(references[0], os.path.join(self.config.eval.image_folder, f'bad/{i}_ref_bad.png'))

#                         plt.imshow(predict[0,0].cpu().numpy())
#                         plt.title(f'predict sum = {predict_num}')
#                         plt.savefig(os.path.join(self.config.eval.image_folder, f'bad/{i}_predict_bad.png'))

#                         plt.imshow(target[0,0].cpu().numpy())
#                         plt.title(f'target sum = {target_num}')
#                         plt.savefig(os.path.join(self.config.eval.image_folder, f'bad/{i}_target_bad.png'))

#                     if abs(target_num - predict_num)<=20:

#                         save_image(queries[0],os.path.join(self.config.eval.image_folder, f'good/{i}_query_good.png'))
#                         # grids = make_grid(references)
#                         save_image(references[0],os.path.join(self.config.eval.image_folder, f'good/{i}_ref_good.png'))

#                         plt.imshow(predict[0,0].cpu().numpy())
#                         plt.title(f'predict sum = {predict_num}')
#                         plt.savefig(os.path.join(self.config.eval.image_folder, f'good/{i}_predict_good.png'))

#                         plt.imshow(target[0,0].cpu().numpy())                        
#                         plt.title(f'target sum = {target_num}')
#                         plt.savefig(os.path.join(self.config.eval.image_folder, f'good/{i}_target_good.png'))


        print(f'mae = {mae_sum/count}')
        print(f'mse = {mse_sum/count}')
