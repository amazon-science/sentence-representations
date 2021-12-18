import os
import sys
import argparse

import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer, AutoConfig

BERT_CLASS = {
    "distilbert": 'distilbert-base-uncased', 
    "bertlarge": 'bert-large-uncased',
    "bertbase": 'bert-base-uncased',
}

SBERT_CLASS = {
    "distilbert": 'distilbert-base-nli-mean-tokens',
    "bertbase": 'bert-base-nli-mean-tokens',
    "bertlarge": 'bert-large-nli-mean-tokens',
}


def get_args(argv):
    parser = argparse.ArgumentParser("Evaluation of the pre-finetuning on NLI")
    parser.add_argument('--s3_respath', type=str, default="s3://dejiao-experiment-meetingsum/evaluation/test/")
    parser.add_argument('--respath', type=str, default="../../resnli/senteval/")
    parser.add_argument('--resprefix', type=str, default='sts')
    parser.add_argument('--path_sts', type=str, default="/home/ec2-user/efs/dejiao-explore/all-datasets/senteval_data/")
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--seed', type=int, default=0)
    # evaluate plain bert or sbert
    parser.add_argument('--use_bert', action='store_true', help="")
    parser.add_argument('--use_sbert', action='store_true', help="")
    parser.add_argument('--bert', type=str, default='distilbert')

    # pretrain params to load the pretrained model
    parser.add_argument('--eval_instance', type=str, default="local")
    parser.add_argument('--pretrained_dir', type=str, default="/home/ec2-user/efs/dejiao-explore/exprres/train_sts/pairsupcon/")
    parser.add_argument('--mode', type=str, default="pairsupcon")
    parser.add_argument('--contrast_type', type=str, default="HardNeg")
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--beta', type=int, default=1)

    parser.add_argument('--pdataname', type=str, default='nli_train_posneg')
    parser.add_argument('--lr', type=float, default=5e-06)
    parser.add_argument('--lr_scale', type=int, default=100)
    parser.add_argument('--p_batchsize', type=int, default=1024)
    parser.add_argument('--pretrain_epoch', type=int, default=3)
    parser.add_argument('--eval_epoch', type=str, default='3')
    parser.add_argument('--pseed', type=int, default=0)
    args = parser.parse_args(argv)
    return args


def get_bert(args, pretrained_path=None):
    config = AutoConfig.from_pretrained(BERT_CLASS[args.bert])
    model = AutoModel.from_pretrained(BERT_CLASS[args.bert], config=config)
    tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[args.bert])
    return model, tokenizer


def get_sbert(model_name):
    sbert = SentenceTransformer(SBERT_CLASS[model_name])
    return sbert


def get_pscbert(bert_model, model_path):
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # tokenizer = AutoTokenizer.from_pretrained(BERT_CLASS[bert_model])
    return model, tokenizer


def setup_path(args):
    """pretrained_model: the path to your stored pretrained model"""
    pres_format = '{}.{}.epoch{}.{}.nli_train_posneg.lr{}.lrscale{}.bs{}.tmp{:.2f}.beta{:.1f}.seed{}'
    pretrained_path = pres_format.format(args.mode, args.contrast_type, args.pretrain_epoch, args.bert, args.lr, args.lr_scale, args.p_batchsize, args.temperature, args.beta, args.pseed)
    
    if args.eval_instance == "sagemaker":
        pretrained_dir = args.pretrained_dir
    else:
        pretrained_dir = os.path.join(args.pretrained_dir, pretrained_path)
        

    if args.eval_epoch == "stsdev":
        pretrained_model_path = os.path.join(pretrained_dir, 'pscbert_stsdev')
        resname = '{}_{}_{}_{}_{}_{}_{}_{}_tmp{}_stsdev_seed{}'.format(args.resprefix, args.bert, args.mode, args.contrast_type, args.lr, args.lr_scale, args.pdataname, args.p_batchsize, args.temperature, args.pseed)
        print("-----EVAL CHECKPOINT TRAINED WITH STS-B AS DEV SET-----")
    else:
        pretrained_model_path = os.path.join(pretrained_dir, f'pscbert_epoch_{args.eval_epoch}')
        resname = '{}_{}_{}_{}_{}_{}_{}_{}_tmp{}_epoch{}_seed{}'.format(args.resprefix, args.bert, args.mode, args.contrast_type, args.lr, args.lr_scale, args.pdataname, args.p_batchsize, args.temperature, args.eval_epoch, args.pseed)
        print(f"-----REGULAR/DEFAULT EVAL-----{pretrained_model_path}")
    return resname, pretrained_model_path


def get_checkpoint(args):    
    if args.use_bert: #evaluate vanilla BERT 
        model, tokenizer = get_bert(args)
        resname = 'sts{}_BERT_{}'.format(args.sts_only, args.bert)
        print("...... loading BERT", args.bert, "resname ", resname)

    elif args.use_sbert: # evaluate SentenceBert 
        resname = 'sts{}_SBERT_{}'.format(args.sts_only, args.bert)
        model = get_sbert(args)
        tokenizer = None
        print("...... loading SBERT", args.bert, "resname ", resname)

    else: # evaluate on pairsupcon or our own pretrained SentenceBert model
        resname, pretrained_model_path = setup_path(args)
        # model, tokenizer = get_bert(args)
        # model.load_state_dict(torch.load(pretrained_model, map_location=args.device)) 
        model, tokenizer = get_pscbert(args.bert, pretrained_model_path)
        print("...... loading ", pretrained_model_path, "to device", args.device, "resname ", resname)
             
    model.to(args.device)
    return model, tokenizer, resname


# def setup_path(args):
#     """pretrained_model: the path to your stored pretrained model"""
#     pres_format = '{}.{}.epoch{}.{}.nli_train_posneg.lr{}.lrscale{}.bs{}.tmp{:.2f}.beta{:.1f}.seed{}/'
#     pretrained_path = pres_format.format(args.mode, args.contrast_type, args.pretrain_epoch, args.bert, args.lr, args.lr_scale, args.p_batchsize, args.temperature, args.beta, args.pseed)
    
#     if args.eval_instance == "sagemaker":
#         pretrained_model_path = args.pretrained_dir
#     else:
#         pretrained_model_path = os.path.join(args.pretrained_dir, pretrained_path)
        

#     if args.eval_epoch == "stsdev":
#         pretrained_model = pretrained_model_path + '/pscbert_stsdev.pt'
#         resname = '{}_{}_{}_{}_{}_{}_{}_{}_tmp{}_stsdev_seed{}'.format(args.resprefix, args.bert, args.mode, args.contrast_type, args.lr, args.lr_scale, args.pdataname, args.p_batchsize, args.temperature, args.pseed)
#         print("-----EVAL CHECKPOINT TRAINED WITH STS-B AS DEV SET-----")

#     else:
#         pretrained_model = pretrained_model_path + '/pscbert_epoch_{}.pt'.format(int(args.eval_epoch))
#         resname = '{}_{}_{}_{}_{}_{}_{}_{}_tmp{}_epoch{}_seed{}'.format(args.resprefix, args.bert, args.mode, args.contrast_type, args.lr, args.lr_scale, args.pdataname, args.p_batchsize, args.temperature, args.eval_epoch, args.pseed)
#         print("-----REGULAR/DEFAULT EVAL-----")
#     return resname, pretrained_model


# def get_checkpoint(args):    
#     if args.use_bert: #evaluate vanilla BERT 
#         model, tokenizer = get_bert(args)
#         resname = 'sts{}_BERT_{}'.format(args.sts_only, args.bert)
#         print("...... loading BERT", args.bert, "resname ", resname)

#     elif args.use_sbert: # evaluate SentenceBert 
#         resname = 'sts{}_SBERT_{}'.format(args.sts_only, args.bert)
#         model = get_sbert(args)
#         tokenizer = None
#         print("...... loading SBERT", args.bert, "resname ", resname)

#     else: # evaluate on pairsupcon or our own pretrained SentenceBert model
#         resname, pretrained_model = setup_path(args)
#         model, tokenizer = get_bert(args)
#         model.load_state_dict(torch.load(pretrained_model, map_location=args.device))  
#         print("...... loading ", pretrained_model, "to device", args.device, "resname ", resname)
             
#     model.to(args.device)
#     return model, tokenizer, resname

