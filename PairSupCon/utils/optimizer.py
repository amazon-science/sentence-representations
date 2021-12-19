import torch 
from transformers import AutoTokenizer, AutoConfig, AutoModel

MODEL_CLASS = {
    "bertlarge": 'bert-large-uncased',
    "bertbase": 'bert-base-uncased',
}


def get_optimizer(model, args):

    if args.mode == "contrastive":
        optimizer = torch.optim.Adam([
            {'params':model.bert.parameters()}, 
            {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    elif args.mode == "classification":
        optimizer = torch.optim.Adam([
            {'params':model.bert.parameters()}, 
            {'params':model.classify_head.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    elif args.mode == "pairsupcon":
        optimizer = torch.optim.Adam([
            {'params':model.bert.parameters()}, 
            {'params':model.classify_head.parameters(), 'lr': args.lr*args.lr_scale},
            {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)

    print("-----mode: {} \n ------optimizer: {}".format(args.mode, optimizer))    
    return optimizer 


def get_bert_config_tokenizer(model_name):
    config = AutoConfig.from_pretrained(MODEL_CLASS[model_name])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CLASS[model_name])
    return config, tokenizer





