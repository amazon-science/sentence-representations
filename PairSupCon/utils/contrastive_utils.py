from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


class HardConLoss(nn.Module):
    def __init__(self, temperature=0.07, contrast_type="HardNeg"):
        super(HardConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_type = contrast_type
        self.eps = 1e-08
        print("-----Contrastive Learning Type: \t", contrast_type)

    def forward(self, features_1, features_2, pairsimi):
        device = (torch.device('cuda') if features_1.is_cuda else torch.device('cpu'))
        batch_size = features_1.shape[0]

        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)

        pairmask = torch.cat([pairsimi, pairsimi], dim=0)
        posmask = (pairmask == 1).detach()
        posmask = posmask.type(torch.int32)

        if self.contrast_type == "Orig":
            Ng = neg.sum(dim=-1)
            loss_pos = (-posmask * torch.log(pos / (Ng+pos))).sum() / posmask.sum()
            return {"instdisc_loss":loss_pos}

        elif self.contrast_type == "HardNeg":
            negimp = neg.log().exp()
            Ng = (negimp*neg).sum(dim = -1) / negimp.mean(dim = -1)
            loss_pos = (-posmask * torch.log(pos / (Ng+pos))).sum() / posmask.sum()
            return {"instdisc_loss":loss_pos}

        else:
            raise Exception("Please specify the contrastive loss, Orig vs. HardNeg.")
        