import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from transformers import BertPreTrainedModel, BertModel


class PairSupConBert(BertPreTrainedModel):
    def __init__(self, config, num_classes=2, feat_dim=128):
        super(PairSupConBert, self).__init__(config)
        print("\n *****Initializing PairSupCon-Bert*****")
    
        self.bert = BertModel(config)
        self.emb_size = self.bert.config.hidden_size
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        
        # Sentence-Bert style input for training the pairwise classification head
        self.classify_head = nn.Linear(3*self.emb_size, self.num_classes, bias=False)

        self.contrast_head = nn.Sequential(
            nn.Linear(self.emb_size, self.emb_size, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, self.feat_dim, bias=False))
      
    
    def forward(self, input_ids, attention_mask, task_type):
    
        if task_type == "evaluate":
            return self.get_mean_embeddings(input_ids, attention_mask)
        else:
            # split inputs w.r.t sentence_1 and sentence_2
            input_ids_1, input_ids_2 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2 = torch.unbind(attention_mask, dim=1) 

            # get the mean embeddings
            bert_output_1 = self.bert.forward(input_ids=input_ids_1, attention_mask=attention_mask_1)
            bert_output_2 = self.bert.forward(input_ids=input_ids_2, attention_mask=attention_mask_2)
            attention_mask_1 = attention_mask_1.unsqueeze(-1)
            attention_mask_2 = attention_mask_2.unsqueeze(-1)
            mean_output_1 = torch.sum(bert_output_1[0]*attention_mask_1, dim=1) / torch.sum(attention_mask_1, dim=1)
            mean_output_2 = torch.sum(bert_output_2[0]*attention_mask_2, dim=1) / torch.sum(attention_mask_2, dim=1)

            if task_type == "classification":
                class_pred = self.classify_pred(mean_output_1, mean_output_2)
                return class_pred
            elif task_type == "contrastive":
                cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
                return cnst_feat1, cnst_feat2
            else:
                # PairSupCon Objective
                class_pred = self.classify_pred(mean_output_1, mean_output_2)
                cnst_feat1, cnst_feat2 = self.contrast_logits(mean_output_1, mean_output_2)
                return class_pred, cnst_feat1, cnst_feat2
       
    
    def contrast_logits(self, embd1, embd2):
        feat1 = F.normalize(self.contrast_head(embd1), dim=1)
        feat2 = F.normalize(self.contrast_head(embd2), dim=1)
        return feat1, feat2

    
    def classify_pred(self, embd1, embd2):
        embeddings = torch.cat([embd1, embd2, torch.abs(embd1-embd2)], 1)
        return self.classify_head(embeddings)
    
    
    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        embeddings = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return embeddings
