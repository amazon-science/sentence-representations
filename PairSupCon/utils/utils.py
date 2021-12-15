import os
import random
import torch
import numpy as np
from tensorboardX import SummaryWriter

def set_global_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def setup_path(args):
    resPath = args.mode
    resPath += f'.{args.contrast_type}'
    resPath += f'.epoch{args.epochs}'
    resPath += f'.{args.bert}'
    resPath += f'.{args.dataname}'
    resPath += f'.lr{args.lr}'
    resPath += f'.lrscale{args.lr_scale}'
    resPath += f'.bs{args.batch_size}'
    resPath += f'.tmp{args.temperature}'
    resPath += f'.beta{args.beta}'
    resPath += f'.seed{args.seed}/'
    resPath = args.resdir + resPath
    print(f'results path: {resPath}')

    tensorboard = SummaryWriter(resPath)
    return resPath, tensorboard


def statistics_log(tensorboard, losses=None, global_step=0):
    print("[{}]-----".format(global_step))
    if losses is not None:
        for key, val in losses.items():
            tensorboard.add_scalar('train/'+key, val.item(), global_step)
            print("{}:\t {:.3f}".format(key, val.item()))
