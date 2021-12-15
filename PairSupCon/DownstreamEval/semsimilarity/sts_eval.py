from __future__ import absolute_import, division, unicode_literals
import os
import sys
import numpy as np
import pandas as pd
import logging
import torch

# Set SENTEVAL PATH
PATH_SENTEVAL = './SentEval'
sys.path.insert(0, PATH_SENTEVAL)
import senteval

def sts_eval(args, transfer_tasks, model, tokenizer, resname):
    # Set up logger and device
    logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

    def prepare(params, samples):
        return 

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        
        features = params.tokenizer.batch_encode_plus(
            sentences, 
            max_length=params['max_length'], 
            return_tensors='pt', 
    #         padding=True, 
            padding = 'max_length',
            truncation=True
        )
        input_ids, attention_mask = features['input_ids'].to(params['device']), features['attention_mask'].to(params['device'])
        bert_output = params.transformer.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings.detach().cpu().numpy()


    # define senteval params
    params_senteval = {'task_path': args.path_sts, 'usepytorch': True, 'kfold': 10}
    params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}
    params_senteval['device'] = args.device
    params_senteval['transformer'] = model
    params_senteval['tokenizer'] = tokenizer
    params_senteval['max_length'] = args.max_length
    
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    
    eval_results = se.eval(transfer_tasks)

    return eval_results





    
    
    
    

