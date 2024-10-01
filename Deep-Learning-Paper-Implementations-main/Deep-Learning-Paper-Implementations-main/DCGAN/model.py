import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim: int = 100, img_channels: int = 3):
        super().__init__()
        # Please refer to figure 1 of the paper.

        # When using batch normalization you have a learnable parameter β which have the same role as bias when not using batch normalization.
        # Adding bias term to Wx will result in a new term when averaging in the batch normalization 
        # algorithm but that term would vanish because the subsequent mean subtraction, 
        # and that why they ignore the biases and this is the purpose of the β learnable parameter.

        self.layers = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 1024, kernel_size = 4, stride = 1, padding = 0, bias = False), # (b, z_dim, 1, 1) -> (b, 1024, 4, 4)
            nn.BatchNorm2d(1024),
            nn.ReLU(True), # inplace
            nn.ConvTranspose2d(1024, 512, kernel_size = 4, stride = 2, padding = 1, bias = False), # (b, 1024, 4, 4) -> (b, 512, 8, 8)
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size = 4, stride = 2, padding = 1, bias = False), # (b, 512, 8, 8) -> (b, 256, 16, 16)
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size = 4, stride = 2, padding = 1, bias = False), # (b, 256, 16, 16) -> (b, 128, 32, 32)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, img_channels, kernel_size = 4, stride = 2, padding = 1, bias = False), # (b, 128, 32, 32) -> (b, 3, 64, 64)
            nn.Tanh() # mentioned in the paper. Section 3.
        )
    
    def forward(self, x):
        return self.layers(x)
    

class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 3, ndf: int = 64):
        super().__init__()

        # conv2D output shape: ((n + (2 * p) - k) / s) + 1
        self.layers = nn.Sequential(
            nn.Conv2d(img_channels, ndf, kernel_size = 4, stride = 2, padding = 1, bias = False), # (b, 3, 64, 64) -> (b, 64, 32, 32)
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2), # mentioned in the paper.
            nn.Conv2d(ndf, ndf * 2, kernel_size = 4, stride = 2, padding = 1, bias = False), # (b, 64, 32, 32) -> (b, 128, 16, 16)
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size = 4, stride = 2, padding = 1, bias = False), # (b, 128, 16, 16) -> (b, 256, 8, 8)
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size = 4, stride = 2, padding = 1, bias = False), # (b, 256, 8, 8) -> (b, 512, 4, 4)
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(ndf * 8, 1, kernel_size = 4, stride = 1, padding = 0, bias = False), # (b, 512, 4, 4) -> (b, 1, 1, 1)
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.layers(x)
    
def test():
    gen = Generator()
    disc = Discriminator()
    x = torch.randn((1, 3, 64, 64))
    assert disc(x).shape == (1, 1, 1, 1)
    z = torch.randn((1, 100, 1, 1))
    assert gen(z).shape == (1, 3, 64, 64)
    print("Success!")

test()
