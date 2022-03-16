import os
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
import numpy as np

import torch
import torch.nn as nn
from models.Transformers import VaSCL_BERT, VaSCL_RoBERTa
from training import VaSCL_Trainer
from dataloader.dataloader import pair_loader
from utils.utils import set_global_random_seed, setup_path
from utils.optimizer import get_optimizer, get_bert_config_tokenizer, MODEL_CLASS
import subprocess
 

def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)
    device = torch.device("cuda:{}".format(args.gpuid[0]) if torch.cuda.is_available() else "cpu")
    print("Let's use GPU", device, args.gpuid)
    
    # dataloader 
    train_loader = pair_loader(args)

    # model and optimizer
    config, tokenizer = get_bert_config_tokenizer(args.bert)
    if 'roberta' in args.bert:
        model = VaSCL_RoBERTa.from_pretrained(MODEL_CLASS[args.bert])
        print(f"*****Loading {args.bert}, {MODEL_CLASS[args.bert]}")
    else:
        model = VaSCL_BERT.from_pretrained(MODEL_CLASS[args.bert])
   
    optimizer = get_optimizer(model, args)

    model = nn.DataParallel(model)
    model.to(device)

    # training
    trainer = VaSCL_Trainer(model, tokenizer, optimizer, train_loader, args)
    trainer.train()

    return None


def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local', choices=["local", "sagemaker"])
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1.")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--logging_step', type=float, default=100, help="")
    parser.add_argument('--resdir', type=str, default='./results')
    parser.add_argument('--s3_ckptdir', type=str, default='YOUR-S3-BUCKET-PATH IF YOU NEED PUSH YOUR CKPT TO S3')

    # Dataset
    parser.add_argument('--datapath', type=str, default='../vascl_data')
    parser.add_argument('--dataname', type=str, default='wiki1m_for_simcse', help="")
    parser.add_argument('--text1', type=str, default='text')
    parser.add_argument('--text2', type=str, default='text')
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--pad_to_max_length', action='store_true', help="")
    # Training parameters
    parser.add_argument('--lr', type=float, default=5e-6, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=100000000)
    parser.add_argument('--devset', type=str, default="sts-b", help="use sts-b as dev set")
    parser.add_argument('--path_sts_data', type=str, default='./senteval', help="use sts-b as dev set")   
    # VaSCL loss
    parser.add_argument('--bert', type=str, default='robertabase', help="")
    parser.add_argument('--temperature', type=float, default=0.05, help="temperature required by contrastive loss")
    parser.add_argument('--topk', type=int, default=16, help=" ")
    parser.add_argument('--eps', type=float, default=15, help=" ")
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None
    return args


if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    if args.train_instance == "sagemaker":
        # copy data from s3 bucket to sagemaker ec2 instance
        args.path_sts_data = os.environ["SM_CHANNEL_DATA"]
        print(f"\n path to sts data {args.path_sts_data} \n")

        run(args)
        # copy the pre-trained checkpoints from sagemaker to s3 bucket
        subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_ckptdir])
    else:
        run(args) # run the code on your local machine


