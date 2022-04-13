import torch
import torch.nn as nn
import torch.nn.functional as F

class cls_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001, model_name=''):
        super(cls_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale
        self.model_name = model_name

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        if self.model_name in ['PointNet','PointNet_Seg'] :
            d = trans_feat.size()[1]
            I = torch.eye(d)[None, :, :]
            if trans_feat.is_cuda:I = I.cuda()
            mat_diff_loss = torch.mean(torch.norm(torch.bmm(trans_feat, trans_feat.transpose(2, 1)) - I, dim=(1, 2)))
            loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return loss