from __future__ import absolute_import, division, unicode_literals
import os 
import sys

import numpy as np
import pandas as pd
import logging
import torch

from configs.configure import get_args, get_checkpoint
from stseval.sts_eval import sts_eval
import subprocess

def parse_results(res, resname):
    
    dfsp = {'model': resname}
    for key, val in res.items():
        if key in ['STS12', 'STS13', 'STS14', 'STS15', 'STS16']:
            dfsp[key] = val['all']['spearman']['all']
        elif key in ['SICKRelatedness', 'STSBenchmark']:
            dfsp[key] = val['test']['spearman'].correlation
        elif key in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'SST5', 'TREC', 'MRPC', 'SICKEntailment']:
            dfsp[key] = val['acc']
        else:
            continue
     
    df = pd.DataFrame([dfsp], index=[0], columns=list(res.keys()))
    df['Avg'] = df.mean(axis=1)
    return df


if __name__ == "__main__":
    args = get_args(sys.argv[1:])
    args.device = torch.device(f"cuda:{args.device_id}") if torch.cuda.is_available() else torch.device("cpu")   
    args.resprefix = "sts"
    
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'SICKRelatedness', 'STSBenchmark']
    try:
        os.makedirs(args.respath, exist_ok = True)
        print("Directory '%s' created successfully" %args.respath)
    except OSError as error:
        print("Directory '%s' can not be created")
    
    model, tokenizer, resname = get_checkpoint(args)

    results = sts_eval(args, transfer_tasks, model, tokenizer)
    np.save(args.respath+resname+'.npy', results)
    
    df = parse_results(results, resname)
    print(df.round(4))

    

        