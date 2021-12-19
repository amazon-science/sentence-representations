import os
import sys
import argparse

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig

BERT_CLASS = {
    "bertbase": 'bert-base-uncased',
    "bertlarge": 'bert-large-uncased',
    "pairsupcon-base": "aws-ai/pairsupcon-bert-base-uncased",
    "pairsupcon-large": "aws-ai/pairsupcon-bert-large-uncased",
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-mean-tokens',
    "bertbase": 'bert-base-nli-mean-tokens',
    "bertlarge": 'bert-large-nli-mean-tokens',
}


def get_args(argv):
    parser = argparse.ArgumentParser("Evaluation of the pre-trained models on various downstream tasks")
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=0)
    # results path and prefix
    parser.add_argument('--respath', type=str, default="../../downstream_evalres/senteval/")
    parser.add_argument('--resprefix', type=str, default='sts')
    # data evaluation configuration
    parser.add_argument('--path_to_sts_data', type=str, default="", help="path to the SentEval data")
    parser.add_argument('--path_to_cluster_data', type=str, default="", help="path to the SentEval data")
    parser.add_argument('--max_length', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    # evaluate the pretrained_model
    parser.add_argument('--pretrained_dir', type=str, default="")
    parser.add_argument('--pretrained_model', type=str, default='PairSupCon', choices=["BERT", "SBERT", "PairSupCon"])
    parser.add_argument('--bert', type=str, default='bertbase')
    args = parser.parse_args(argv)
    return args


def get_bert(args):
    config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
    model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
    tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
    return model, tokenizer


def get_sbert(model_name):
    sbert = SentenceTransformer(SBERT_CLASS[model_name])
    return sbert



def get_checkpoint(args): 
    
    if args.pretrained_model in ["BERT", "PairSupCon"]: #evaluate vanilla BERT or PairSupCon
        model, tokenizer = get_bert(args)
        resname = 'sts_{}_{}'.format(args.pretrained_model, args.bert)
        print("...... loading BERT", args.bert, "resname ", resname)

    elif args.pretrained_model == "SBERT": # evaluate SentenceBert 
        resname = 'sts{}_SBERT_{}'.format(args.sts_only, args.bert)
        model = get_sbert(args)
        tokenizer = None
        print("...... loading SBERT", args.bert, "resname ", resname)

    else: 
        raise Exception("please specify the pretrained model you want to evaluate")
             
    model.to(args.device)
    return model, tokenizer, resname


