import torch
import torch.nn as nn
from einops.einops import rearrange
from .backbone import ResNet_8_2
from .utils.position_encoding import PositionEncodingSine
from .xoftr_module import LocalFeatureTransformer, FineProcess


class XoFTR_Pretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Misc
        self.config = config
        self.patch_size = config["pretrain_patch_size"]

        # Modules
        self.backbone = ResNet_8_2(config['resnet'])
        self.pos_encoding = PositionEncodingSine(config['coarse']['d_model'])
        self.loftr_coarse = LocalFeatureTransformer(config['coarse'])
        self.fine_process = FineProcess(config)
        self.mask_token_f = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][0], 1, 1))
        self.mask_token_m = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][1], 1, 1))
        self.mask_token_c = nn.Parameter(torch.zeros(1, config['resnet']["block_dims"][2], 1, 1))
        self.out_proj = nn.Linear(config['resnet']["block_dims"][0], 4)
    
        torch.nn.init.normal_(self.mask_token_f, std=.02)
        torch.nn.init.normal_(self.mask_token_m, std=.02)
        torch.nn.init.normal_(self.mask_token_c, std=.02)

    def upsample_mae_mask(self, mae_mask, scale):
        assert len(mae_mask.shape) == 2
        p = int(mae_mask.shape[1] ** .5)
        return mae_mask.reshape(-1, p, p).repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def upsample_mask(self, mask, scale):
        return mask.repeat_interleave(scale, axis=1).repeat_interleave(scale, axis=2)
    
    def mask_layer(self, feat, mae_mask, mae_mask_scale, mask=None, mask_scale=None, mask_token=None):
        """ Mask the feature map and replace with trainable inpu tokens if available
        Args:
            feat (torch.Tensor): [N, C, H, W]
            mae_mask (torch.Tensor): (N, L) mask for masked image modeling
            mae_mask_scale (int): the scale of layer to mae mask
            mask (torch.Generator): mask for padded input image
            mask_scale (int): the scale of layer to mask (mask is created on course scale)
            mask_token (torch.Tensor): [1, C, 1, 1] learnable mae mask token
        Returns:
            feat (torch.Tensor): [N, C, H, W]
        """ 
        mae_mask = self.upsample_mae_mask(mae_mask, mae_mask_scale)
        mae_mask = mae_mask.unsqueeze(1).type_as(feat)
        if mask is not None:
            mask =  self.upsample_mask(mask, mask_scale)
            mask = mask.unsqueeze(1).type_as(feat)
            mae_mask = mask * mae_mask
        feat = feat * (1. - mae_mask)
        if mask_token is not None:
            mask_token = mask_token.repeat(feat.shape[0], 1, feat.shape[2], feat.shape[3])
            feat += mask_token * mae_mask
        return feat
        

    def forward(self, data):
        """ 
        Update:
            data (dict): {
                'image0': (torch.Tensor): (N, 1, H, W)
                'image1': (torch.Tensor): (N, 1, H, W)
                'mask0'(optional) : (torch.Tensor): (N, H, W) '0' indicates a padded position
                'mask1'(optional) : (torch.Tensor): (N, H, W)
            }
        """
        # 1. Local Feature CNN
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:]
        })

        image0 = data["image0_norm"] if "image0_norm" in data else data["image0"]
        image1 = data["image1_norm"] if "image1_norm" in data else data["image1"]

        mask0 = mask1 = None  # mask fro madded images
        if 'mask0' in data:
            mask0, mask1 = data['mask0'], data['mask1']

        # mask input images
        image0 = self.mask_layer(image0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size,
                                mask=mask0,
                                mask_scale=8)
        image1 = self.mask_layer(image1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size,
                                mask=mask1,
                                mask_scale=8)
        data.update({"masked_image0":image0.clone().detach().cpu(),
                     "masked_image1":image1.clone().detach().cpu()})

        if data['hw0_i'] == data['hw1_i']:  # faster & better BN convergence
            feats_c, feats_m, feats_f = self.backbone(torch.cat([image0, image1], dim=0))
            (feat_c0, feat_c1) = feats_c.split(data['bs'])
            (feat_m0, feat_m1) = feats_m.split(data['bs'])
            (feat_f0, feat_f1) = feats_f.split(data['bs'])
        else:  # handle different input shapes
            feat_c0, feat_m0, feat_f0 = self.backbone(image0)
            feat_c1, feat_m1, feat_f1 = self.backbone(image1)
        
        # mask output layers of backbone and replace with trainable token
        feat_c0 = self.mask_layer(feat_c0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 8,
                                mask=mask0,
                                mask_scale=1,
                                mask_token=self.mask_token_c)
        feat_c1 = self.mask_layer(feat_c1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 8,
                                mask=mask1,
                                mask_scale=1,
                                mask_token=self.mask_token_c)
        feat_m0 = self.mask_layer(feat_m0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 4,
                                mask=mask0,
                                mask_scale=2,
                                mask_token=self.mask_token_m)
        feat_m1 = self.mask_layer(feat_m1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 4,
                                mask=mask1,
                                mask_scale=2,
                                mask_token=self.mask_token_m)
        feat_f0 = self.mask_layer(feat_f0,
                                data["mae_mask0"],
                                mae_mask_scale=self.patch_size // 2,
                                mask=mask0,
                                mask_scale=4,
                                mask_token=self.mask_token_f)
        feat_f1 = self.mask_layer(feat_f1,
                                data["mae_mask1"],
                                mae_mask_scale=self.patch_size // 2,
                                mask=mask1,
                                mask_scale=4,
                                mask_token=self.mask_token_f)
        data.update({
            'hw0_c': feat_c0.shape[2:], 'hw1_c': feat_c1.shape[2:],
            'hw0_m': feat_m0.shape[2:], 'hw1_m': feat_m1.shape[2:],
            'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]
        })

        # save coarse features for fine matching module
        feat_c0_pre, feat_c1_pre = feat_c0.clone(), feat_c1.clone()

        # 2. Coarse-level loftr module
        # add featmap with positional encoding, then flatten it to sequence [N, HW, C]
        feat_c0 = rearrange(self.pos_encoding(feat_c0), 'n c h w -> n (h w) c')
        feat_c1 = rearrange(self.pos_encoding(feat_c1), 'n c h w -> n (h w) c')

        mask_c0 = mask_c1 = None  # mask is useful in training
        if 'mask0' in data:
            mask_c0, mask_c1 = data['mask0'].flatten(-2), data['mask1'].flatten(-2)
        feat_c0, feat_c1 = self.loftr_coarse(feat_c0, feat_c1, mask_c0, mask_c1)

        # 3. Fine-level maching module as decoder
        # generate window locations from mae mask to reconstruct
        mae_mask_c0 = self.upsample_mae_mask( data["mae_mask0"],
                                self.patch_size // 8)
        if mask0 is not None:
            mae_mask_c0 = mae_mask_c0 * mask0.type_as(mae_mask_c0)
            
        mae_mask_c1 = self.upsample_mae_mask( data["mae_mask1"],
                                self.patch_size // 8)
        if mask1 is not None:
            mae_mask_c1 = mae_mask_c1 * mask1.type_as(mae_mask_c1)
        
        mae_mask_c = torch.logical_or(mae_mask_c0, mae_mask_c1)

        b_ids, i_ids = mae_mask_c.flatten(1).nonzero(as_tuple=True)
        j_ids = i_ids

        # b_ids, i_ids and j_ids are masked location for both images
        # ids_image0 and ids_image1 determines which indeces belogs to which image
        ids_image0 = mae_mask_c0.flatten(1)[b_ids, i_ids]
        ids_image1 = mae_mask_c1.flatten(1)[b_ids, j_ids]

        data.update({'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids,
                     'ids_image0': ids_image0==1, 'ids_image1': ids_image1==1})


        # fine level matching module
        feat_f0_unfold, feat_f1_unfold = self.fine_process( feat_f0, feat_f1,
                                                            feat_m0, feat_m1,
                                                            feat_c0, feat_c1,
                                                            feat_c0_pre, feat_c1_pre,
                                                            data)

        # output projection 5x5 window to 10x10 window
        pred0 = self.out_proj(feat_f0_unfold)
        pred1 = self.out_proj(feat_f1_unfold)

        data.update({"pred0":pred0, "pred1": pred1})


    def load_state_dict(self, state_dict, *args, **kwargs):
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        return super().load_state_dict(state_dict, *args, **kwargs)
