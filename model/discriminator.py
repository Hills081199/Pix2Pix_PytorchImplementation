import torch
import torch.nn as nn
from model.components import DownBlock

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = DownBlock(in_channels=6, out_channels=64, norm_layer=nn.BatchNorm2d)
        self.down_2 = DownBlock(in_channels=64, out_channels=128, norm_layer=nn.BatchNorm2d) 
        self.down_3 = DownBlock(in_channels=128, out_channels=256, norm_layer=nn.BatchNorm2d)
        self.down_4 = DownBlock(in_channels=256, out_channels=512, norm_layer=nn.BatchNorm2d)
        self.down_5 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.down_3(x)
        x = self.down_4(x)
        x = self.down_5(x)

        return x
    
# if __name__ == '__main__':
#     D_input_x = torch.randn(1, 3, 512, 512)
#     D_input_y = torch.randn(1, 3, 512, 512)
#     D = Discriminator()
#     print(D(D_input_x, D_input_y).shape)