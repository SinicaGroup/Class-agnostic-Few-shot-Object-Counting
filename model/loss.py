import torch.nn as nn
from pytorch_msssim import ssim

def my_loss(predicted_density_map, ground_truth_density_map,config):
    L2loss = nn.MSELoss(reduction='mean').to(config.device)
    Standard_L2_loss = L2loss(predicted_density_map, ground_truth_density_map)
    SSIM_loss = 1*config.train.batch_size - ssim(predicted_density_map,
                         ground_truth_density_map, data_range=1.0, size_average=True)

   
    Final_loss = Standard_L2_loss + config.train.ssim_loss * SSIM_loss
    
    return Final_loss, SSIM_loss