import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset


class PairSamples(Dataset):
    def __init__(self, train_x1, train_x2):
        assert len(train_x1) == len(train_x2)
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        
        
    def __len__(self):
        return len(self.train_x1)

    def __getitem__(self, idx):
        return {'text1': self.train_x1[idx], 'text2': self.train_x2[idx]}


def pair_loader(args):
    print(f"load data: {os.path.join(args.datapath, args.dataname+'.csv')}" )
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+'.csv'))
    train_text1 = train_data[args.text1].fillna('.').values
    train_text2 = train_data[args.text2].fillna('.').values

    train_dataset = PairSamples(train_text1, train_text2)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader





