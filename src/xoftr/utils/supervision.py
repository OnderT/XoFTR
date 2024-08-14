from math import log
from loguru import logger

import torch
import torch.nn.functional as F
from einops import repeat
from kornia.utils import create_meshgrid
from einops.einops import rearrange
from .geometry import warp_kpts, warp_kpts_fine
from kornia.geometry.epipolar import fundamental_from_projections, normalize_transformation

##############  ↓  Coarse-Level supervision  ↓  ##############


@torch.no_grad()
def mask_pts_at_padded_regions(grid_pt, mask):
    """For megadepth dataset, zero-padding exists in images"""
    mask = repeat(mask, 'n h w -> n (h w) c', c=2)
    grid_pt[~mask.bool()] = 0
    return grid_pt


@torch.no_grad()
def spvs_coarse(data, config):
    """
    Update:
        data (dict): {
            "conf_matrix_gt": [N, hw0, hw1],
            'spv_b_ids': [M]
            'spv_i_ids': [M]
            'spv_j_ids': [M]
            'spv_w_pt0_i': [N, hw0, 2], in original image resolution
            'spv_pt1_i': [N, hw1, 2], in original image resolution
        }
        
    NOTE:
        - for scannet dataset, there're 3 kinds of resolution {i, c, f}
        - for megadepth dataset, there're 4 kinds of resolution {i, i_resize, c, f}
    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['XOFTR']['RESOLUTION'][0]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0 * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1 * grid_pt1_c

    # mask padded region to (0, 0), so no need to manually mask conf_matrix_gt
    if 'mask0' in data:
        grid_pt0_i = mask_pts_at_padded_regions(grid_pt0_i, data['mask0'])
        grid_pt1_i = mask_pts_at_padded_regions(grid_pt1_i, data['mask1'])

    # warp kpts bi-directionally and resize them to coarse-level resolution
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    valid_mask0, w_pt0_i = warp_kpts(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'])
    valid_mask1, w_pt1_i = warp_kpts(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'])
    w_pt0_i[~valid_mask0] = 0
    w_pt1_i[~valid_mask1] = 0
    w_pt0_c = w_pt0_i / scale1
    w_pt1_c = w_pt1_i / scale0

    # 3. nearest neighbor
    w_pt0_c_round = w_pt0_c[:, :, :].round().long()
    nearest_index1 = w_pt0_c_round[..., 0] + w_pt0_c_round[..., 1] * w1
    w_pt1_c_round = w_pt1_c[:, :, :].round().long()
    nearest_index0 = w_pt1_c_round[..., 0] + w_pt1_c_round[..., 1] * w0

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_c_round, w1, h1)] = 0
    nearest_index0[out_bound_mask(w_pt1_c_round, w0, h0)] = 0

    arange_1 = torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    arange_0 = torch.arange(h0*w0, device=device)[None].repeat(N, 1)
    arange_1[nearest_index1 == 0] = 0
    arange_0[nearest_index0 == 0] = 0
    arange_b = torch.arange(N, device=device).unsqueeze(1)

    # 4. construct a gt conf_matrix
    conf_matrix_gt = torch.zeros(N, h0*w0, h1*w1, device=device)
    conf_matrix_gt[arange_b, arange_1, nearest_index1] = 1
    conf_matrix_gt[arange_b, nearest_index0, arange_0] = 1
    conf_matrix_gt[:, 0, 0] = False

    b_ids, i_ids, j_ids = conf_matrix_gt.nonzero(as_tuple=True)

    data.update({'conf_matrix_gt': conf_matrix_gt})

    # 5. save coarse matches(gt) for training fine level
    if len(b_ids) == 0:
        logger.warning(f"No groundtruth coarse match found for: {data['pair_names']}")
        # this won't affect fine-level loss calculation
        b_ids = torch.tensor([0], device=device)
        i_ids = torch.tensor([0], device=device)
        j_ids = torch.tensor([0], device=device)

    data.update({
        'spv_b_ids': b_ids,
        'spv_i_ids': i_ids,
        'spv_j_ids': j_ids
    })

    # 6. save intermediate results (for fast fine-level computation)
    data.update({
        'spv_w_pt0_i': w_pt0_i,
        'spv_pt1_i': grid_pt1_i
    })

def compute_supervision_coarse(data, config):
    assert len(set(data['dataset_name'])) == 1, "Do not support mixed datasets training!"
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_coarse(data, config)
    else:
        raise ValueError(f'Unknown data source: {data_source}')


##############  ↓  Fine-Level supervision  ↓  ##############

def compute_supervision_fine(data, config):
    data_source = data['dataset_name'][0]
    if data_source.lower() in ['scannet', 'megadepth']:
        spvs_fine(data, config)
    else:
        raise NotImplementedError
    
@torch.no_grad()
def create_2d_gaussian_kernel(kernel_size, sigma, device):
    """
    Create a 2D Gaussian kernel.
    
    Args:
        kernel_size (int): Size of the kernel (both width and height).
        sigma (float): Standard deviation of the Gaussian distribution.
        
    Returns:
        torch.Tensor: 2D Gaussian kernel.
    """
    kernel = torch.arange(kernel_size, dtype=torch.float32, device=device) - (kernel_size - 1) / 2
    kernel = torch.exp(-kernel**2 / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    
    # Outer product to get a 2D kernel
    kernel = torch.outer(kernel, kernel)
    
    return kernel

@torch.no_grad()
def create_conf_prob(points, h0, w0, h1, w1, kernel_size = 5, sigma=1):
    """
    Place a gaussian kernel in sim matrix for warped points 
      
    Args:
        data (dict): {
            points: (torch.Tensor): (N, L, 2), warped rounded key points
            h0, w0, h1, w1: (int), windows sizes
            kernel_size: (int), kernel size for the gaussian
            sigma: (float), sigma value for gaussian
        }
    """
    B = points.shape[0]
    impulses = torch.zeros(B, h0 * w0, h1, w1, device=points.device)

    # Extract the row and column indices
    row_indices = points[:, :, 1]
    col_indices = points[:, :, 0]

    # Set the corresponding locations in the target tensor to 1
    impulses[torch.arange(B, device=points.device).view(B, 1, 1),
            torch.arange(h0 * w0, device=points.device).view(1, h0 * w0, 1),
            row_indices.unsqueeze(-1), col_indices.unsqueeze(-1)] = 1
    # mask 0,0 point
    impulses[:,:,0,0] = 0

    # Create the Gaussian kernel
    gaussian_kernel = create_2d_gaussian_kernel(kernel_size, sigma=sigma, device=points.device)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)

    # Create distributions at the points
    conf_prob = F.conv2d(impulses.view(-1,1,h1,w1), gaussian_kernel, padding=kernel_size//2).view(-1, h0*w0, h1*w1)

    return conf_prob

@torch.no_grad()
def spvs_fine(data, config):
    """
    Args:
        data (dict): {
            'b_ids': [M]
            'i_ids': [M]
            'j_ids': [M]
        }
        
    Update:
        data (dict): {
            conf_matrix_f_gt: [N, W_f^2, W_f^2], in original image resolution
            }

    """
    # 1. misc
    device = data['image0'].device
    N, _, H0, W0 = data['image0'].shape
    _, _, H1, W1 = data['image1'].shape
    scale = config['XOFTR']['RESOLUTION'][1]
    scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
    scale1 = scale * data['scale1'][:, None] if 'scale1' in data else scale
    h0, w0, h1, w1 = map(lambda x: x // scale, [H0, W0, H1, W1])
    scale_f_c = config['XOFTR']['RESOLUTION'][0] // config['XOFTR']['RESOLUTION'][1]
    W_f = config['XOFTR']['FINE_WINDOW_SIZE']
    # 2. get coarse prediction
    b_ids, i_ids, j_ids = data['b_ids'], data['i_ids'], data['j_ids']

    if len(b_ids) == 0:
        data.update({"conf_matrix_f_gt": torch.zeros(1,W_f*W_f,W_f*W_f, device=device)})
        return

    # 2. warp grids
    # create kpts in meshgrid and resize them to image resolution
    grid_pt0_c = create_meshgrid(h0, w0, False, device).repeat(N, 1, 1, 1)#.reshape(1, h0*w0, 2).repeat(N, 1, 1)    # [N, hw, 2]
    grid_pt0_i = scale0[:,None,...] * grid_pt0_c
    grid_pt1_c = create_meshgrid(h1, w1, False, device).repeat(N, 1, 1, 1)#.reshape(1, h1*w1, 2).repeat(N, 1, 1)
    grid_pt1_i = scale1[:,None,...] * grid_pt1_c
    
    # unfold (crop windows) all local windows
    stride_f = data['hw0_f'][0] // data['hw0_c'][0]

    grid_pt0_i = rearrange(grid_pt0_i, 'n h w c -> n c h w')
    grid_pt0_i = F.unfold(grid_pt0_i, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
    grid_pt0_i = rearrange(grid_pt0_i, 'n (c ww) l -> n l ww c', ww=W_f**2)
    grid_pt0_i = grid_pt0_i[b_ids, i_ids]

    grid_pt1_i = rearrange(grid_pt1_i, 'n h w c -> n c h w')
    grid_pt1_i = F.unfold(grid_pt1_i, kernel_size=(W_f, W_f), stride=stride_f, padding=W_f//2)
    grid_pt1_i = rearrange(grid_pt1_i, 'n (c ww) l -> n l ww c', ww=W_f**2)
    grid_pt1_i = grid_pt1_i[b_ids, j_ids]

    # warp kpts bi-directionally and resize them to fine-level resolution
    # (no depth consistency check
    # (unhandled edge case: points with 0-depth will be warped to the left-up corner)
    _, w_pt0_i = warp_kpts_fine(grid_pt0_i, data['depth0'], data['depth1'], data['T_0to1'], data['K0'], data['K1'], b_ids)
    _, w_pt1_i = warp_kpts_fine(grid_pt1_i, data['depth1'], data['depth0'], data['T_1to0'], data['K1'], data['K0'], b_ids)
    w_pt0_f = w_pt0_i / scale1[b_ids]
    w_pt1_f = w_pt1_i / scale0[b_ids]

    mkpts0_c_scaled_to_f = torch.stack(
        [i_ids % data['hw0_c'][1], i_ids // data['hw0_c'][1]],
        dim=1) * scale_f_c - W_f//2
    mkpts1_c_scaled_to_f = torch.stack(
        [j_ids % data['hw1_c'][1], j_ids // data['hw1_c'][1]],
        dim=1) * scale_f_c - W_f//2
    
    w_pt0_f = w_pt0_f - mkpts1_c_scaled_to_f[:,None,:]
    w_pt1_f = w_pt1_f - mkpts0_c_scaled_to_f[:,None,:]

    # 3. check if mutual nearest neighbor
    w_pt0_f_round = w_pt0_f[:, :, :].round().long()
    w_pt1_f_round = w_pt1_f[:, :, :].round().long()
    M = w_pt0_f.shape[0]

    nearest_index1 = w_pt0_f_round[..., 0] + w_pt0_f_round[..., 1] * W_f
    nearest_index0 = w_pt1_f_round[..., 0] + w_pt1_f_round[..., 1] * W_f

    # corner case: out of boundary
    def out_bound_mask(pt, w, h):
        return (pt[..., 0] < 0) + (pt[..., 0] >= w) + (pt[..., 1] < 0) + (pt[..., 1] >= h)
    nearest_index1[out_bound_mask(w_pt0_f_round, W_f, W_f)] = 0
    nearest_index0[out_bound_mask(w_pt1_f_round, W_f, W_f)] = 0

    loop_back = torch.stack([nearest_index0[_b][_i] for _b, _i in enumerate(nearest_index1)], dim=0)
    correct_0to1 = loop_back == torch.arange(W_f*W_f, device=device)[None].repeat(M, 1)
    correct_0to1[:, 0] = False  # ignore the top-left corner

    # 4. construct a gt conf_matrix
    conf_matrix_f_gt = torch.zeros(M, W_f*W_f, W_f*W_f, device=device)
    b_ids, i_ids = torch.where(correct_0to1 != 0)
    j_ids = nearest_index1[b_ids, i_ids]
    conf_matrix_f_gt[b_ids, i_ids, j_ids] = 1

    data.update({"conf_matrix_f_gt": conf_matrix_f_gt})


