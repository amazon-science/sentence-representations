import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

shorttext_clustering_datapath = "your-path"


CLUSTER_DATASETS = {
    "agnews":(shorttext_clustering_datapath + 'agnews.csv', 4, 'text', 'label'),
    "searchsnippets":(shorttext_clustering_datapath + 'searchsnippets.csv', 8, 'text', 'label'),
    "stackoverflow":(shorttext_clustering_datapath + 'stackoverflow.csv', 20, 'text', 'label'),
    "biomedical":(shorttext_clustering_datapath + 'biomedical.csv', 20, 'text', 'label'),
    "tweet":(shorttext_clustering_datapath + 'tweet89.csv', 89, 'text', 'label'),
    "googleT":(shorttext_clustering_datapath + 'googlenews_T.csv', 152, 'text', 'label'),
    "googleS":(shorttext_clustering_datapath + 'googlenews_S.csv', 152, 'text', 'label'),
    "googleTS":(shorttext_clustering_datapath + 'googlenews_TS.csv', 152, 'text', 'label'),
}


class TextClustering(Dataset):
    def __init__(self, train_x, train_y):
        assert len(train_x) == len(train_y)
        self.train_x = train_x
        self.train_y = train_y

    def __len__(self):
        return len(self.train_x)

    def __getitem__(self, idx):
        return {'text': self.train_x[idx], 'label': self.train_y[idx]}
      

def cluster_data_loader(datapath, text, label, batch_size):
    train_data = pd.read_csv(datapath)
    train_text = train_data[text].fillna('.').values
    train_label = train_data[label].astype(int).values
    print(len(train_text), len(train_label))

    train_dataset = TextClustering(train_text, train_label)
    train_loader = util_data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader


