import torch
import torch.nn as nn
from math import log2
import torch.nn.functional as F

class WSConv2D(nn.Module): # paper section 4.1
    def __init__(self, in_chans, out_chans, kernel_size = 3, stride = 1, padding = 1, gain = 2): # (n x n) conv (f x f) -> ((n + 2p - f) / s, (n + 2p - f) / s)
        super().__init__()
        self.conv = nn.Conv2d(in_chans, out_chans, kernel_size, stride, padding, bias = False) # params chosen to ensure 'same' convolution
        self.bias = nn.Parameter(torch.zeros(out_chans))
        self.ws = (gain / (kernel_size * kernel_size * in_chans)) ** 0.5 # formula

        nn.init.normal_(self.conv.weight.data)
        nn.init.zeros_(self.bias) 

    def forward(self, x):
        # self.conv(x * self.ws) shape -> (batch_size, out_chans, n, n)
        # self.bias shape -> (out_cha, ). To add we need to broadcast it to (1, out_chans, 1, 1)
        return self.conv(x * self.ws) + self.bias.view(1, self.bias.shape[0], 1, 1)

class PixelNorm(nn.Module): # paper section 4.2
    def __init__(self, eps = 1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, x):
        return x / torch.sqrt(torch.mean(x**2, dim = 1, keepdim = True) + self.eps)
    
class ConvBlock(nn.Module): #Implementing a single block. Table 2 of the paper
    def __init__(self, in_chans, out_chans, use_pixel_norm = True):
        super().__init__()
        self.conv1 = WSConv2D(in_chans, out_chans)
        self.leaky = nn.LeakyReLU(0.2)
        self.conv2 = WSConv2D(out_chans, out_chans)
        self.pn = PixelNorm()
        self.use_pn = use_pixel_norm # This is because pixel norm is only used in the generator
    
    def forward(self, x):
        x = self.leaky(self.conv1(x))
        x = self.pn(x) if self.use_pn is True else x
        x = self.leaky(self.conv2(x))
        x = self.pn(x) if self.use_pn is True else x
        return x

class WSConvTrans2D(nn.Module): # Latent vector 512x1x1 -> 512x4x4 with Weight Scaling.
    def __init__(self, in_chans, out_chans, kernel_size = 4, stride = 1, gain = 2):
        super().__init__()
        self.trans_conv = nn.ConvTranspose2d(in_chans, out_chans, kernel_size = kernel_size, stride = stride, bias = False) # parameters from the equations given in TransposeConv2D page of torch.
        self.bias = nn.Parameter(torch.zeros(out_chans))
        self.ws = (gain / (kernel_size * kernel_size * in_chans)) ** 0.5 # formula

        nn.init.normal_(self.trans_conv.weight.data)
        nn.init.zeros_(self.bias)

    def forward(self, x):
        return self.trans_conv(x * self.ws) + self.bias.view(1, self.bias.shape[0], 1, 1) 

factors = [1, 1, 1, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16, 1 / 32] # 9 blocks as given in paper

class Generator(nn.Module):
    def __init__(self, z_dim, in_chans, img_chans = 3):
        super().__init__()
        self.initial = nn.Sequential(
            WSConvTrans2D(z_dim, in_chans),
            nn.LeakyReLU(0.2),
            WSConv2D(in_chans, in_chans),
            nn.LeakyReLU(0.2),
            PixelNorm()
        )
        self.initial_rgb = WSConv2D(in_chans, img_chans, 1, 1, 0) # same size, img_channels
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([self.initial_rgb]) # 9 rgb_layers, 8 progressive blocks, because we have separate initial block above

        for i in range(len(factors) - 1):
            in_c_chans = int(in_chans * factors[i])
            out_c_chans = int(in_chans * factors[i + 1])
            self.prog_blocks.append(ConvBlock(in_c_chans, out_c_chans))
            self.rgb_layers.append(WSConv2D(out_c_chans, img_chans, 1, 1, 0))
        
    def fade_in(self, alpha, upscaled, generated):
        return torch.tanh((alpha * generated) + ((1 - alpha) * upscaled))
        
    def forward(self, x, alpha, steps):
        out = self.initial(x)

        if steps == 0:
            return self.initial_rgb(out)
        
        for i in range(steps):
            upscaled = F.interpolate(out, scale_factor = 2, mode = "nearest")
            out = self.prog_blocks[i](upscaled)
        
        final_upscaled = self.rgb_layers[steps - 1](upscaled)
        # to rgb
        final_generated = self.rgb_layers[steps](out)   # Please look into figure 2 of the paper!

        return self.fade_in(alpha, final_upscaled, final_generated)
    
class Discriminator(nn.Module):
    def __init__(self, in_chans, img_chans = 3): # in_chans is 512 acc. to paper
        super().__init__()
        self.prog_blocks, self.rgb_layers = nn.ModuleList([]), nn.ModuleList([])
        self.leaky = nn.LeakyReLU(0.2)

        for i in range(len(factors) - 1, 0, -1):
            in_c_chans = (int)(in_chans * factors[i])
            out_c_chans = (int)(in_chans * factors[i - 1])
            self.prog_blocks.append(ConvBlock(in_c_chans, out_c_chans, use_pixel_norm = False)) 
            self.rgb_layers.append(WSConv2D(img_chans, in_c_chans, 1, 1, 0))
        
        #rgb layer for the last block 
        self.final_rgb = WSConv2D(img_chans, in_chans, 1, 1, 0)
        self.rgb_layers.append(self.final_rgb)

        self.avg_pool = nn.AvgPool2d(kernel_size = 2, stride = 2)

        # last block in discriminator
        self.final_block = nn.Sequential(
            #after mini batch stddev
            WSConv2D(in_chans + 1, in_chans, 3, 1, 1),
            nn.LeakyReLU(0.2),
            WSConv2D(in_chans, in_chans, 4, 1, 0),
            nn.LeakyReLU(0.2),
            WSConv2D(in_chans, 1, 1, 1, 0)            
        )
    
    def fade_in(self, alpha, downscaled, out):
        return ((alpha * out) + ((1 - alpha) * downscaled))
    
    def minibatch_std(self, x):
        std = torch.std(x, dim = 0).mean().repeat(x.shape[0], 1, x.shape[2], x.shape[3]) # shape -> (1, ). repeat because you want to add this across channel dimension.
        return torch.cat([x, std], dim = 1)
    
    def forward(self, x, alpha, steps): # steps == 0, signify last block and so on..
        cur_step = len(self.prog_blocks) - steps

        out = self.leaky(self.rgb_layers[cur_step](x))
        if steps == 0:
            out = self.minibatch_std(out)
            return self.final_block(out).view(x.shape[0], -1) # (x.shape[0], 1, 1, 1) -> (x.shape[0], 1) ------- x.shape[0] -> batch size
        
        # from rgb. Please look fig 2 in paper!
        downscaled = self.leaky(self.rgb_layers[cur_step + 1](self.avg_pool(x)))
        out = self.avg_pool(self.prog_blocks[cur_step](out))

        out = self.fade_in(alpha, downscaled, out)

        for step in range(cur_step + 1, len(self.prog_blocks)):
            out = self.prog_blocks[step](out)
            out = self.avg_pool(out)
        
        out = self.minibatch_std(out)
        return self.final_block(out).view(x.shape[0], -1)
    

# test!

if __name__ == "__main__":
    z_dim = 100
    in_chans = 256
    gen = Generator(z_dim, in_chans)
    critic = Discriminator(in_chans)

    for img_size in [4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        num_steps = (int)(log2(img_size / 4))
        input = torch.randn((1, z_dim, 1, 1))
        gen_output = gen(input, 0.5, num_steps)
        assert gen_output.shape == (1, 3, img_size, img_size)
        disc_output = critic(gen_output, 0.5, num_steps)
        assert disc_output.shape == (1, 1)
        print(f"Success! At img size: {img_size}")
