from __future__ import absolute_import, division, unicode_literals
import os
import sys

import time
import numpy as np
import pandas as pd
import torch

from configs.configure import get_args, get_checkpoint
from clustering.clustering_eval import clustering_eval


if __name__ == "__main__":

    args = get_args(sys.argv[1:])
    args.device = torch.device(f"cuda:{args.device_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.eval_instance == "sagemaker":
        args.pretrained_dir = os.environ["SM_CHANNEL_PRETRAIN"]

    args.resprefix = "cluster"
    datanames = ["appen_human", "appen_asr", "agnews", "searchsnippets", "stackoverflow", "biomedical", "tweet", "googleT", "googleS", "googleTS"]
    
    
    
    args.eval_epoch = epoch
        
    model, tokenizer, resname = get_checkpoint(args)

    df = clustering_eval(model, tokenizer, datanames, resname, args)

    df.to_csv(args.respath+resname+'.csv')
        
