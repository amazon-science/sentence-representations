import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset


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


