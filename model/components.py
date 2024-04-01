import torch.nn as nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm_layer=None):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = None
        if norm_layer:
            self.norm = norm_layer(out_channels)
        
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        
        x = self.leaky_relu(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, norm_layer=None, dropout=False):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
        self.norm = None
        self.dropout = None
        if norm_layer:
            self.norm = norm_layer(out_channels)
        
        if dropout:
            self.dropout = nn.Dropout2d(p=0.3, inplace=True)
    
    def forward(self, x):
        x = self.deconv(x)
        if self.norm:
            x = self.norm(x)
        
        x = self.leaky_relu(x)
        if self.dropout:
            x = self.dropout(x)

        return x