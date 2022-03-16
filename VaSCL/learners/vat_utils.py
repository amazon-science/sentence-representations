import time
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F

@contextlib.contextmanager
def _disable_tracking_bn_stats(model):
    def switch_attr(m):
        if hasattr(m, 'track_running_stats'):
            m.track_running_stats ^= True
    model.apply(switch_attr)
    yield
    model.apply(switch_attr)

def _l2_normalize(d, attention_mask=None):
    if attention_mask != None:
        attention_mask = attention_mask.unsqueeze(-1)
        d *= attention_mask
    d_reshaped = d.view(d.shape[0], -1, *(1 for _ in range(d.dim() - 2)))
    d /= torch.norm(d_reshaped, dim=1, keepdim=True) + 1e-8
#     print("_l2_normalize, BEFORE:{} \t AFTER:{}".format(d.size(), d_reshaped.size()))
    return d

def _emb_norm(emb):
    e_reshaped = emb.view(emb.shape[0], -1, *(1 for _ in range(emb.dim() - 2)))
    enorm = torch.norm(e_reshaped, dim=1, keepdim=False) + 1e-8
#     print("BEFORE:{} \t AFTER:{}".format(emb.size(), e_reshaped.size()))
#     print("enorm:{}, {}".format(enorm.size(), enorm[:10]))
    return enorm

    
class VaSCL_Pturb(nn.Module):
    def __init__(self, xi=0.1, eps=1, ip=1, uni_criterion=None, bi_criterion=None):
        """VaSCL_Pturb on Transformer embeddings
            :param xi: hyperparameter of VaSCL_Pturb (default: 10.0)
            :param eps: hyperparameter of VaSCL_Pturb (default: 1.0)
            :param ip: iteration times of computing adv noise (default: 1)
        """
        super(VaSCL_Pturb, self).__init__()
        self.xi = xi
        self.eps = eps
        self.ip = ip
        self.delta = 1e-08
        
        self.uni_criterion = uni_criterion
        self.bi_criterion = bi_criterion
        print("\n VaSCL_Pturb on embeddings, xi:{}, eps:{} \n".format(xi, eps))

    def forward(self, model, inputs, hard_indices):
#         print(inputs.size(), "\n", _emb_norm(inputs)[:5])
        with torch.no_grad():
            cnst = model.module.contrast_logits(inputs)

        # prepare random unit tensor
        d = torch.rand(inputs.shape).sub(0.5).to(inputs.device)
        d = _l2_normalize(d)
        
        with _disable_tracking_bn_stats(model):
            # calc adversarial direction
            for _ in range(self.ip):
                d.requires_grad_()
                cnst_hat = model.module.contrast_logits(inputs+self.xi*d)
                
                adv_cnst = self.uni_criterion(cnst, cnst_hat, hard_indices)
                adv_distance = adv_cnst['lds_loss']

                adv_distance.backward(retain_graph=True)
                d = _l2_normalize(d.grad)
                model.zero_grad()

        cnst_hat = model.module.contrast_logits(inputs+self.eps*d)
        adv_cnst = self.bi_criterion(cnst, cnst_hat, hard_indices)
        return adv_cnst
    
    
    

    

