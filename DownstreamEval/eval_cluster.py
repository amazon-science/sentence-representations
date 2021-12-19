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
    args.resprefix = "cluster"
    
    datanames = ["agnews", "searchsnippets", "stackoverflow", "biomedical", "tweet", "googleT", "googleS", "googleTS"]
    # datanames = ["agnews", "searchsnippets"]

    CLUSTER_DATASETS = {
        "agnews":(args.path_to_cluster_data + 'agnews.csv', 4, 'text', 'label'),
        "searchsnippets":(args.path_to_cluster_data + 'searchsnippets.csv', 8, 'text', 'label'),
        "stackoverflow":(args.path_to_cluster_data + 'stackoverflow.csv', 20, 'text', 'label'),
        "biomedical":(args.path_to_cluster_data + 'biomedical.csv', 20, 'text', 'label'),
        "tweet":(args.path_to_cluster_data + 'tweet89.csv', 89, 'text', 'label'),
        "googleT":(args.path_to_cluster_data + 'googlenews_T.csv', 152, 'text', 'label'),
        "googleS":(args.path_to_cluster_data + 'googlenews_S.csv', 152, 'text', 'label'),
        "googleTS":(args.path_to_cluster_data + 'googlenews_TS.csv', 152, 'text', 'label'),
    }
    
        
    model, tokenizer, resname = get_checkpoint(args)

    df = clustering_eval(model, tokenizer, datanames, resname, args, CLUSTER_DATASETS)
    df.to_csv(args.respath+resname+'.csv')
    
    dfs = df.groupby(["data"])["ACC", "NMI", "AMI", "ARI"].agg(["mean", "std"]).round(4)
    print(dfs)
        
