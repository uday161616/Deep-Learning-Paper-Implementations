"""
Discriminators take as input, an image of size 256x256 and output a tensor of size 30x30.
Each neuron (value) of the output tensor holds the classification result for a 70x70 area of the input image. 
Usually, discriminator of GANs output one value to indicate the classification result of the input image. 
By returning a tensor of size 30x30, the discriminator checks if every 70x70 area — these areas overlap each other — of the input image seems real or fake. 
Doing so is equivalent to manually select each of these 70x70 areas and have the discriminator examine them iteratively. 
Finally the classification result on the whole image is the average of classification results on the 30x30 values.
https://towardsdatascience.com/overview-of-cyclegan-architecture-and-training-afee31612a2f
"""

import torch
import torch.nn as nn
from typing import List

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 4, stride = stride, padding = 1, padding_mode = 'reflect'), # mentioned in the paper
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace = True)  
        )
    
    def forward(self, x):
        return self.block(x)

class Discriminator(nn.Module): # input shape generally 3 x 256 x 256
    def __init__(self, img_channels: int = 3, features: List[int] = [64, 128, 256, 512]) -> None:
        super().__init__()
        # writing first layer separately because it doesn't include instance norm and leaky relu. (3 x 256 x 256) -> (64 x 128 x 128)
        self.initial = nn.Conv2d(
            img_channels,
            features[0],
            kernel_size = 4,
            stride = 2,
            padding = 1,
            padding_mode = 'reflect'
        )
        layers = []
        in_channels = 64
        for feature in features[1:]:
            layers.append(ConvBlock(in_channels, feature, stride = 1 if feature == features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size = 4, stride = 1, padding = 1, padding_mode = 'reflect')) # (512 x 31 x 31) -> (1 x 30 x 30)
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.initial(x)
        return self.model(x)

def test():
    x = torch.randn((3, 256, 256))
    disc = Discriminator()
    output = disc(x)
    assert output.shape == (1, 30, 30)
    print(output.shape)
    # print(output.max(), output.min())
    # print(output)

if __name__ == "__main__":
    test()

