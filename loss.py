import torch
import torch.nn as nn

class DLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss()
    
    def forward(self, fake_pred, real_pred):
        fake_target = torch.zeros_like(fake_pred)
        real_target = torch.ones_like(real_pred)
        loss = (self.loss(fake_pred, fake_target) + self.loss(real_pred, real_target)) / 2
        return loss

class GLoss(nn.Module):
    def __init__(self, alpha=100):
        super().__init__()
        self.alpha = alpha
        self.loss = nn.BCEWithLogitsLoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, fake, real, fake_pred):
        fake_target = torch.ones_like(fake_pred)
        loss = self.loss(fake_pred, fake_target) + self.alpha* self.l1(fake, real)
        return loss
    