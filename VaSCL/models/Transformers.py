import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertPreTrainedModel, BertModel, RobertaPreTrainedModel, RobertaModel


class VaSCL_RoBERTa(RobertaPreTrainedModel):
    def __init__(self, config):
        super(VaSCL_RoBERTa, self).__init__(config)
        print("-----Initializing VaSCL_RoBERTa-----")
        self.roberta = RobertaModel(config)
        self.emb_size = self.roberta.config.hidden_size
        self.feat_dim = 128

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))

    def forward(self, input_ids, attention_mask, topk=16, task_type="train"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 
            
            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            inner_prod = torch.mm(mean_output_1, mean_output_1.t().contiguous())

            # estimate the neighborhood of input example
            batch_size = input_ids.shape[0]
            mask = torch.eye(batch_size, dtype=torch.bool).to(mean_output_1.device)
            inner_prod_neg = inner_prod.masked_select(~mask).view(batch_size, -1)
            topk_inner, hard_indices = torch.topk(inner_prod_neg, k=topk, dim=-1)
            # print(f"\n embeddings:{mean_output_1.size()}\t inner_prod:{inner_prod_neg.size()}\t topk_inner:{topk_inner.size()}\t{hard_indices.size()}\n")

            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            return mean_output_1, hard_indices, cnst_feat1, cnst_feat2
    
    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.roberta.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output
     
    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else: 
            return feat1


class VaSCL_BERT(BertPreTrainedModel):
    def __init__(self, config):
        super(VaSCL_BERT, self).__init__(config)
        print("-----Initializing VaSCL_BERT-----")
        self.bert = BertModel(config)
        self.emb_size = self.bert.config.hidden_size
        self.feat_dim = 128

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))

    def forward(self, input_ids, attention_mask, topk=16, task_type="train"):
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        else:
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1)

            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            inner_prod = torch.mm(mean_output_1, mean_output_1.t().contiguous())

            # estimate the neighborhood of input example
            batch_size = input_ids.shape[0]
            mask = torch.eye(batch_size, dtype=torch.bool).to(mean_output_1.device)
            inner_prod_neg = inner_prod.masked_select(~mask).view(batch_size, -1)
            topk_inner, hard_indices_unidir = torch.topk(inner_prod_neg, k=topk, dim=-1)

            cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
            return mean_output_1, hard_indices_unidir, cnst_feat1, cnst_feat2

    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def contrast_logits(self, embd1, embd2=None):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        if embd2 != None:
            feat2 = F.normalize(self.contrast_head(embd2), dim=1)
            return feat1, feat2
        else:
            return feat1
    

    
    