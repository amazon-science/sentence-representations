import os
import pandas as pd
import torch.utils.data as util_data
from torch.utils.data import Dataset

class PairSamples(Dataset):
    def __init__(self, train_x1, train_x2, pairsimi):
        assert len(pairsimi) == len(train_x1) == len(train_x2)
        self.train_x1 = train_x1
        self.train_x2 = train_x2
        self.pairsimi = pairsimi
        
    def __len__(self):
        return len(self.pairsimi)

    def __getitem__(self, idx):
        return {'text1': self.train_x1[idx], 'text2': self.train_x2[idx], 'pairsimi': self.pairsimi[idx]}

def pair_loader(args):
    train_data = pd.read_csv(os.path.join(args.datapath, args.dataname+'.csv'))
    
    # assume each input pair is named as (sentence1, sentence2)
    train_text1 = train_data[args.text+'1'].fillna('.').values
    train_text2 = train_data[args.text+'2'].fillna('.').values
    pairsimi = train_data[args.pairsimi].astype(int).values

    train_dataset = PairSamples(train_text1, train_text2, pairsimi)
    train_loader = util_data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    return train_loader

