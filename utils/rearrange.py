from einops import rearrange
import numpy as np
import torch
from torch import nn

if __name__ == '__main__':
    img = torch.randn(1, 3, 256, 256)
    patch_size = 32
    img_h, img_w = img.shape[-2:]
    patch_h, patch_w = (patch_size, patch_size)
    num_patches = (img_h * img_w) / patch_size * patch_size
    patch_dim = num_patches * patch_size * patch_size


    patches = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_h, p2=patch_w)
    # nn.LayerNorm(patch_dim)

    print(patches.shape)
