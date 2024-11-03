import torch.nn as nn
import torch
class AoDLoss(nn.Module):
    def __init__(self):
        super(AoDLoss, self).__init__()
        
    def forward(self, AOD_true, AOD_pred):
        N = AOD_true.size(0)
        
        sum_true = torch.sum(AOD_true)
        sum_pred = torch.sum(AOD_pred)
        
        sum_true_pred = torch.sum(AOD_true * AOD_pred)
        
        square_sum_true = torch.sum(AOD_true ** 2)
        square_sum_pred = torch.sum(AOD_pred ** 2)
        
        num = N * sum_true_pred - sum_true * sum_pred
        den = torch.sqrt((N * square_sum_true - sum_true ** 2) * (N * square_sum_pred - sum_pred ** 2))
        
        pearson_r = num / (den + 1e-8)  # Adding small epsilon for numerical stability
        
        return pearson_r  