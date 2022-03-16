import torch
from transformers import AutoModel, AutoTokenizer, AutoConfig

MODEL_CLASS = {
    "robertabase": 'roberta-base',
    "robertalarge": 'roberta-large',
    "bertlarge": 'bert-large-uncased',
    "bertbase": 'bert-base-uncased',
}

def get_optimizer(model, args):
    if 'roberta' in args.bert:
        optimizer = torch.optim.Adam([
        {'params':model.roberta.parameters()},
        {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)
    else:
        optimizer = torch.optim.Adam([
            {'params':model.bert.parameters()},
            {'params':model.contrast_head.parameters(), 'lr': args.lr*args.lr_scale}], lr=args.lr)

    return optimizer 
    

def get_bert_config_tokenizer(model_name):
    config = AutoConfig.from_pretrained(MODEL_CLASS[model_name])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CLASS[model_name])
    return config, tokenizer


def get_bert(model_name):
    config = AutoConfig.from_pretrained(MODEL_CLASS[model_name])
    auto_model = AutoModel.from_pretrained(MODEL_CLASS[model_name], config=config)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_CLASS[model_name])
    # auto_model.resize_token_embeddings(len(tokenizer))
    return auto_model, tokenizer









