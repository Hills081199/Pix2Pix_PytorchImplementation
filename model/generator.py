from model.components import UpBlock, DownBlock
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.down_2 = DownBlock(in_channels=64, out_channels=128, norm_layer=nn.BatchNorm2d)
        self.down_3 = DownBlock(in_channels=128, out_channels=256, norm_layer=nn.BatchNorm2d)
        self.down_4 = DownBlock(in_channels=256, out_channels=512, norm_layer=nn.BatchNorm2d)
        self.down_5 = DownBlock(in_channels=512, out_channels=512, norm_layer=nn.BatchNorm2d)
        self.down_6 = DownBlock(in_channels=512, out_channels=512, norm_layer=nn.BatchNorm2d)
        self.down_7 = DownBlock(in_channels=512, out_channels=512, norm_layer=nn.BatchNorm2d)
        self.down_8 = DownBlock(in_channels=512, out_channels=512)

        self.up_8 = UpBlock(in_channels=512, out_channels=512, norm_layer=nn.BatchNorm2d, dropout=True)
        self.up_7 = UpBlock(in_channels=1024, out_channels=512, norm_layer=nn.BatchNorm2d, dropout=True)
        self.up_6 = UpBlock(in_channels=1024, out_channels=512, norm_layer=nn.BatchNorm2d, dropout=True)
        self.up_5 = UpBlock(in_channels=1024, norm_layer=nn.BatchNorm2d, out_channels=512)
        self.up_4 = UpBlock(in_channels=1024, norm_layer=nn.BatchNorm2d, out_channels=256)
        self.up_3 = UpBlock(in_channels=512, norm_layer=nn.BatchNorm2d, out_channels=128)
        self.up_2 = UpBlock(in_channels=256, norm_layer=nn.BatchNorm2d, out_channels=64)
        self.up_1 = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x):
        d1 = self.down_1(x)
        d2 = self.down_2(d1)
        d3 = self.down_3(d2)
        d4 = self.down_4(d3)
        d5 = self.down_5(d4)
        d6 = self.down_6(d5)
        d7 = self.down_7(d6)
        d8 = self.down_8(d7)            #(512, 2, 2)    
        
        u8 = self.up_8(d8)              #(512, 4, 4)
        u8 = torch.cat([u8, d7], dim=1)
        u7 = self.up_7(u8)
        u7 = torch.cat([u7, d6], dim=1)
        u6 = self.up_6(u7)
        u6 = torch.cat([u6, d5], dim=1)
        u5 = self.up_5(u6)
        u5 = torch.cat([u5, d4], dim=1)
        u4 = self.up_4(u5)
        u4 = torch.cat([u4, d3], dim=1)
        u3 = self.up_3(u4)
        u3 = torch.cat([u3, d2], dim=1)
        u2 = self.up_2(u3)
        u2 = torch.cat([u2, d1], dim=1)
        u1 = self.up_1(u2)              #(3, 512, 512)
        
        return torch.tanh(u1)


# #fortest
# if __name__ == '__main__':
#     import torch
#     G_input = torch.randn(1,3,512,512)
#     G = Generator()
#     #G(G_input)
#     print(G(G_input).shape)