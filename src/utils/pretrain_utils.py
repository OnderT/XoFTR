import torch 
import torch.nn as nn
from einops.einops import rearrange
import torch.nn.functional as F

def generate_random_masks(batch, patch_size, mask_ratio, generator=None, margins=[0,0,0,0]):
    mae_mask0 = _gen_random_mask(batch['image0'], patch_size, mask_ratio, generator, margins=margins)
    mae_mask1 = _gen_random_mask(batch['image1'], patch_size, mask_ratio, generator, margins=margins)
    batch.update({"mae_mask0" : mae_mask0, "mae_mask1": mae_mask1})

def _gen_random_mask(image, patch_size, mask_ratio, generator=None, margins=[0, 0, 0, 0]): 
    """ Random mask generator
    Args:
        image (torch.Tensor): [N, C, H, W]
        patch_size (int)
        mask_ratio (float)
        generator (torch.Generator): RNG to create the same random masks for validation  
        margins [float, float, float, float]: unused part for masking (up bottom left right)
    Returns:
        mask (torch.Tensor): (N, L)
    """ 
    N = image.shape[0]
    l = (image.shape[2] // patch_size)
    L = l ** 2
    len_keep = int(L *  (1 - mask_ratio * (1 - sum(margins))))

    margins = [int(margin * l) for margin in margins]

    noise = torch.rand(N, l, l, device=image.device, generator=generator)
    if margins[0] > 0 : noise[:,:margins[0],:] = 0
    if margins[1] > 0 : noise[:,-margins[1]:,:] = 0
    if margins[2] > 0 : noise[:,:,:margins[2]] = 0
    if margins[3] > 0 : noise[:,:,-margins[3]:] = 0
    noise = noise.flatten(1)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # generate the binary mask: 0 is keep 1 is remove
    mask = torch.ones([N, L], device=image.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask

def patchify(data):
    """ Split images into small overlapped patches
    Args:
        data (dict):{
            'image0_norm' (torch.Tensor): [N, C, H, W] normalized image,
            'image1_norm' (torch.Tensor): [N, C, H, W] normalized image,
    Returns:
        image0 (torch.Tensor): [N, K, W_f**2, -1] (K: num of windows)
        image1 (torch.Tensor): [N, K, W_f**2, -1] (K: num of windows)
    """
    stride = data['hw0_i'][0] // data['hw0_c'][0]
    scale = data['hw0_i'][0] // data['hw0_f'][0]
    W_f = data["W_f"]
    kernel_size = [int(W_f*scale), int(W_f*scale)]
    padding = kernel_size[0]//2 -1 if kernel_size[0] % 2 == 0 else kernel_size[0]//2

    image0 = data["image0_norm"] if "image0_norm" in data else data["image0"]
    image1 = data["image1_norm"] if "image1_norm" in data else data["image1"]
    
    image0 = F.unfold(image0, kernel_size=kernel_size, stride=stride, padding=padding)
    image0 = rearrange(image0, 'n (c h p w q) l -> n l h w p q c', h=W_f, w=W_f, p=scale, q=scale)
    image0 = image0.flatten(4)
    image0 = image0.reshape(*image0.shape[:2], W_f**2, -1)
    
    image1 = F.unfold(image1, kernel_size=kernel_size, stride=stride, padding=padding)
    image1 = rearrange(image1, 'n (c h p w q) l -> n l h w p q c', h=W_f, w=W_f, p=scale, q=scale)
    image1 = image1.flatten(4)
    image1 = image1.reshape(*image1.shape[:2], W_f**2, -1)

    return image0, image1

def get_target(data):
    """Create target patches for mae"""
    target0, target1 = patchify(data)
    data.update({"target0":target0, "target1":target1})


