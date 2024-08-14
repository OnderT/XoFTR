import torch
import torch.nn as nn
import torch.nn.functional as F

class XoFTRLossPretrain(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.W_f = config["xoftr"]['fine_window_size']
    
    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}

        pred0, pred1 = data["pred0"], data["pred1"]
        target0, target1 = data["target0"], data["target1"]
        target0 = target0[[data['b_ids'], data['i_ids']]]
        target1 = target1[[data['b_ids'], data['j_ids']]]
        
        # get correct indices
        pred0 = pred0[data["ids_image0"]]
        pred1 = pred1[data["ids_image1"]]
        target0 = target0[data["ids_image0"]]
        target1 = target1[data["ids_image1"]]
        
        loss0 = (pred0 - target0)**2
        loss1 = (pred1 - target1)**2
        loss = loss0.mean() + loss1.mean()
        
        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
