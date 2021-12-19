import os 
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import argparse
import torch
import torch.nn as nn

from models.Transformers import PairSupConBert
from training import PairSupConTrainer
from dataloader.dataloader import pair_loader
from utils.utils import set_global_random_seed, setup_path
from utils.optimizer import get_optimizer, MODEL_CLASS, get_bert_config_tokenizer
import subprocess
    
def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = torch.cuda.device_count()
    print("\t {} GPUs available to use!".format(device_id))

    # dataloader
    train_loader = pair_loader(args)
    
    config, tokenizer = get_bert_config_tokenizer(args.bert)
    model = PairSupConBert.from_pretrained(MODEL_CLASS[args.bert])
    optimizer = get_optimizer(model, args)

    model = nn.DataParallel(model)
    model.to(device)
    
    # set up the trainer
    trainer = PairSupConTrainer(model, tokenizer, optimizer, train_loader, args)
    trainer.train()
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_instance', type=str, default='local')
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--resdir', type=str, default='./results')
    parser.add_argument('--logging_step', type=int, default=250, help="")
    parser.add_argument('--dev_set', default="None", help="use sts-b as dev set or not", choices=["None", "sts"])  
    parser.add_argument('--path_sts_data', type=str, default='/home/ec2-user/efs/dejiao-explore/all-datasets/senteval_data/', help="use sts-b as dev set")
    parser.add_argument('--s3_ckptdir', type=str, default='/home/ec2-user/efs/dejiao-explore/all-datasets/senteval_data/', help="s3path for ckpts")  
    # Dataset
    parser.add_argument('--datapath', type=str, default='../datasets/NLI/')
    parser.add_argument('--dataname', type=str, default='nli_pairsupcon.csv', help="")
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--pairsimi', type=str, default='pairsimi')
    # Training parameters
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=5e-06, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max_iter', type=int, default=100000000)
    # Contrastive learning
    parser.add_argument('--mode', type=str, default='pairsupcon', help="")
    parser.add_argument('--bert', type=str, default='bertbase', choices=["bertbase", "bertlarge"], help="")
    parser.add_argument('--contrast_type', type=str, default="HardNeg")
    parser.add_argument('--feat_dim', type=int, default=128, help="dimension of the projected features for instance discrimination loss")
    parser.add_argument('--temperature', type=float, default=0.05, help="temperature required by contrastive loss")
    parser.add_argument('--beta', type=float, default=1, help=" ")
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None
    return args

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

   
    if args.training_instance == "sagemaker":
         # set the input data path if use sts-b as dev set, as ec2 instance cannot read most the data formats included in SentEval 
         # by default, we do not use the sts-b as the dev set for PairSupCon
        args.path_sts_data = os.environ["SM_CHANNEL_DATA"]
        print(f"\n path to sts data {args.path_sts_data} \n")
        run(args)
        
        # upload the saved checkpoints to s3 folder 
        subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_ckptdir])
        
    else:
        run(args)




    