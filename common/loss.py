import torch
import torch.nn as nn
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        return 1 - dice

class myLoss:
    def __init__(self):
        super(myLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        self.dice_loss = DiceLoss()

    def __call__(self, out,batch_dict,contrast_loss=0.1):
        loss_dict = {} 

        gt_mask = batch_dict['anno_mask']
        diceloss_out = self.dice_loss(out, gt_mask.float())
        bceloss_out = self.bce_loss(out, gt_mask.float())

        loss_dict['total_loss'] = diceloss_out + bceloss_out
        return loss_dict

def compute_loss(pred, target, edge_pred, edge_target, lambda_edge=0.1):
    dice_loss = 1 - (2 * (pred * target).sum(dim=(1, 2, 3)) + 1) / ((pred + target).sum(dim=(1, 2, 3)) + 1)
    bce_loss = F.binary_cross_entropy_with_logits(pred, target)

    edge_loss = F.binary_cross_entropy_with_logits(edge_pred, edge_target)

    return dice_loss + bce_loss + lambda_edge * edge_loss

