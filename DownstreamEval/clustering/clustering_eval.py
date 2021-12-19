from __future__ import absolute_import, division, unicode_literals
import os
import sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import numpy as np
import pandas as pd
from sklearn import cluster

import torch
from .metric import Confusion
from .dataloader import cluster_data_loader

PATH_CONFIG = '../'
sys.path.insert(0, PATH_CONFIG)
from configs.utils import set_global_random_seed
from configs.configure import get_args, get_checkpoint


def get_embeddings(transformer, tokenizer, train_loader, device):
    t0 = time.time()
    for i, batch in enumerate(train_loader):
        text, label = batch['text'], batch['label']
        if tokenizer is None:
            corpus_embeddings = transformer.encode(text)
        else:
            features = tokenizer.batch_encode_plus(text, max_length=64, return_tensors='pt', padding='max_length', truncation=True)
            input_ids, attention_mask = features['input_ids'].to(device), features['attention_mask'].to(device)
            bert_output = transformer.forward(input_ids=input_ids, attention_mask=attention_mask)
            corpus_embeddings = torch.sum(bert_output[0]*attention_mask.unsqueeze(-1), dim=1) / torch.sum(attention_mask.unsqueeze(-1), dim=1)
            corpus_embeddings = corpus_embeddings.detach().cpu().numpy()
            
        if i == 0:
            all_labels = label
            all_embeddings = corpus_embeddings
        else:
            all_labels = torch.cat((all_labels, label), dim=0)
            all_embeddings = np.concatenate((all_embeddings, corpus_embeddings), axis=0)
    t1 = time.time()
    print("embeddings:", all_embeddings.shape, "time consumed:",t1-t0)
    return all_labels.cpu().numpy(), all_embeddings


def clustering_single_trial(y_true, embeddings=None, num_classes=10, random_state=0):
    """"Evaluate the embeddings using KMeans"""
    kmeans = cluster.KMeans(n_clusters=num_classes, random_state=random_state)
    kmeans.fit(embeddings)
    y_pred = kmeans.labels_.astype(np.int)
    confusion = Confusion(num_classes)
    confusion.add(torch.tensor(y_pred), torch.tensor(y_true))
    confusion.optimal_assignment(num_classes)
    return confusion.acc(), confusion.clusterscores()


def clustering_eval(bert, tokenizer, datanames, resname, args, CLUSTER_DATASETS):
    clures, index, count = [], [], 0
    for data in datanames:
        datapath, num_classes, text, label = CLUSTER_DATASETS[data]
        train_loader = cluster_data_loader(datapath, text, label, args.batch_size)
        y_true, embeddings = get_embeddings(bert, tokenizer, train_loader, args.device)

        for trial in range(10):
            set_global_random_seed(trial)
            acc, scores = clustering_single_trial(y_true, embeddings, num_classes, random_state=trial)
            scores.update({"data":data, "trial":trial, "ACC":acc})
            clures.append(scores)
            index.append(count)
            count += 1
    df = pd.DataFrame(clures, index=index, columns=list(scores.keys()))
    dfstats = df.groupby(["data"])["ACC", "NMI", "AMI", "ARI"].agg(["mean", "std"]).round(4)
    print("--clustering--\n", dfstats)
    return df


    