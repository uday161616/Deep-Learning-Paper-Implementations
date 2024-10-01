import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down = True, use_act = True, **kwargs):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode = 'reflect', **kwargs) 
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace = True) if use_act else nn.Identity()
        )
    
    def forward(self, x):
        return self.block(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, in_channels, kernel_size = 3, padding = 1),
            ConvBlock(in_channels, in_channels, kernel_size = 3, use_act = False, padding = 1) # mentioned in other implementations
        )
    
    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels = 3, features = 64, num_residuals = 9):
        super().__init__()
        self.initial = ConvBlock(img_channels, features, kernel_size = 7, padding = 3, stride = 1)
        self.down_blocks = nn.Sequential(
            ConvBlock(features, features * 2, kernel_size = 3, stride = 2, padding = 1),
            ConvBlock(features * 2, features * 4, kernel_size = 3, stride = 2, padding = 1)
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(features * 4) for _ in range(num_residuals)])
        self.up_blocks = nn.Sequential(
            ConvBlock(features * 4, features * 2, down = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
            ConvBlock(features * 2, features, down = False, kernel_size = 3, stride = 2, padding = 1, output_padding = 1)
        )
        self.final = nn.Conv2d(features, img_channels, kernel_size = 7, stride = 1, padding = 3, padding_mode = 'reflect') # we'll not use instance norm and relu for last layer
        self.tanh = nn.Tanh()
    
    def forward(self, x):
        x = self.initial(x)
        x = self.down_blocks(x)
        x = self.residual_blocks(x)
        x = self.up_blocks(x)
        x = self.tanh(self.final(x))
        return x
    

def test():
    x = torch.randn((3, 256, 256))
    g = Generator()
    output = g(x)
    assert output.shape == (3, 256, 256)
    print(output.shape)

if __name__ == "__main__":
    test()
