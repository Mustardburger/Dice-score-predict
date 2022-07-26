import torch
import torch.nn as nn
import numpy as np

class MSE(nn.Module):

    def __init__(self):
        super().__init__()
    
    def forward(self, dsc_true, dsc_pred):
        score = torch.sqrt(((dsc_true - dsc_pred)**2).sum())
        return score
