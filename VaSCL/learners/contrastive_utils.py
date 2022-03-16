from __future__ import print_function
import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.05, topk=16):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.topk = topk
        print(f"\n PosConLoss with temperature={temperature}, \t topk={topk}\n")

    def forward(self, features_1, features_2):
        device = features_1.device
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
        
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)

        Ng = neg.sum(dim=-1)
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        return {"loss":loss_pos}
            

class VaSCL_NUniDir(nn.Module):
    def __init__(self, temperature=0.05):
        super(VaSCL_NUniDir, self).__init__()
        self.temperature = temperature
        print(f"\n VaSCL_NUniDir \n")

    def forward(self, features_1, features_2, hard_indices=None):
        device = features_1.device
        batch_size = features_1.shape[0]
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = ~mask

        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        neg = torch.exp(torch.mm(features_2, features_1.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(batch_size, -1)

        hard_mask = torch.zeros_like(neg, dtype=torch.int32).to(device)
        hard_mask = hard_mask.scatter_(1, hard_indices, 1) > 0
        hardneg = neg.masked_select(hard_mask).view(batch_size, -1)

        Ng = hardneg.sum(dim=-1) 
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        return {"lds_loss":loss_pos}
    

class VaSCL_NBiDir(nn.Module):
    def __init__(self, temperature=0.05):
        super(VaSCL_NBiDir, self).__init__()
        self.temperature = temperature
        print(f"\n VaSCL_NBiDir \n")

    def forward(self, features_1, features_2, hard_indices=None):
        device = features_1.device
        batch_size = features_1.shape[0]
        features= torch.cat([features_1, features_2], dim=0)
        mask = torch.eye(batch_size, dtype=torch.bool).to(device)
        mask = mask.repeat(2, 2)
        mask = ~mask
         
        pos = torch.exp(torch.sum(features_1*features_2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)
        neg = torch.exp(torch.mm(features, features.t().contiguous()) / self.temperature)
        neg = neg.masked_select(mask).view(2*batch_size, -1)
        
        hard_mask = torch.zeros(int(neg.shape[0]/2), int(neg.shape[1]/2), dtype=torch.int32).to(device)
        hard_mask = hard_mask.scatter_(1, hard_indices, 1) > 0
        hard_mask = hard_mask.repeat(2, 2)
        hardneg = neg.masked_select(hard_mask).view(2*batch_size, -1)

        Ng = hardneg.sum(dim=-1)  
        loss_pos = (- torch.log(pos / (Ng+pos))).mean()
        return {"lds_loss":loss_pos}