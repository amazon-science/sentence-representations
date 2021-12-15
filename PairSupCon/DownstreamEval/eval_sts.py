from __future__ import absolute_import, division, unicode_literals
import os 
import sys

import numpy as np
import pandas as pd
import logging
import torch

from configs.configure import get_args, get_checkpoint
from semsimilarity.sts_eval import sts_eval
import subprocess


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args.device = torch.device(f"cuda:{args.device_id}") if torch.cuda.is_available() else torch.device("cpu")
    
    if args.eval_instance == "sagemaker":
        args.pretrained_dir = os.environ["SM_CHANNEL_PRETRAIN"]
        args.path_sts = os.environ["SM_CHANNEL_DATA"]
        
    args.resprefix = "sts"
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICKRelatedness', 'STSBenchmark']
    try:
        os.makedirs(args.respath, exist_ok = True)
        print("Directory '%s' created successfully" %args.respath)
    except OSError as error:
        print("Directory '%s' can not be created")
    
    
    args.eval_epoch = epoch
    
    model, tokenizer, resname = get_checkpoint(args)

    results = sts_eval(args, transfer_tasks, model, tokenizer, resname)

    np.save(args.respath+resname+'.npy', results)
    print(results, results.keys())
        
        
    if args.eval_instance == "sagemaker":
        subprocess.run(["aws", "s3", "cp", "--recursive", args.respath, args.s3_respath,])

