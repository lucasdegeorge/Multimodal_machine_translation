#%% 
import torch.nn as nn
import torch
from Multimodal_Attention import *
import json

with open("parameters.json", 'r') as f:
    parameters = json.load(f)
    device = parameters["device"]


class DecoderLayer(nn.Module):   # for text only, we use the classical attention layer
    def __init__(self):
        super().__init__()

        d_model = parameters["d_model"]
        n_heads = parameters["n_heads"]
        dim_feedforward = parameters["dim_feedforward"]
        dropout = parameters["dropout"]

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)   
        self.attn_2 = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )  
        
    def forward(self, x, e_output, tgt_mask, tgt_key_padding_mask, e_key_padding_mask):
        x_1 = self.norm_1(x) 
        x = x + self.dropout_1(
            self.attn_1(x_1, x_1, x_1, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)[0]
        ) 
        x_2 = self.norm_2(x) 
        attn_output, attn_weights = self.attn_2(x_2, e_output, e_output, key_padding_mask=e_key_padding_mask)
        x = x + self.dropout_2(attn_output)  
        x_3 = self.norm_3(x) 
        x = x + self.dropout_3(self.ffn(x_3)) 

        return x, attn_weights



class Multimodal_DecoderLayer(nn.Module):   # we use the multimodal attention layer. See file Multimodal_attention.py
    def __init__(self):
        super().__init__()

        d_model = parameters["d_model"]
        n_heads = parameters["n_heads"]
        dim_feedforward = parameters["dim_feedforward"]
        dropout = parameters["dropout"]

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model, n_heads, dropout, batch_first=True)
        self.attn_2 = MultimodalAttention()

        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )  
        
    def forward(self, x, e_output, i_output, tgt_mask, tgt_key_padding_mask, e_key_padding_mask, ei_key_padding_mask):
        ei_output = torch.cat((e_output, i_output), 1)

        x_1 = self.norm_1(x)
        x = x + self.dropout_1(
            self.attn_1(x_1, x_1, x_1, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask)[0]
        )
        x_2 = self.norm_2(x)
        attn_output, attn_weights_e, attn_weights_i = self.attn_2(x_2, e_output, i_output, ei_output, e_output, i_output, ei_output, e_padding_mask=e_key_padding_mask, ei_padding_mask=ei_key_padding_mask)
        x = x + self.dropout_2(attn_output)
        x_3 = self.norm_3(x)
        x = x + self.dropout_3(self.ffn(x_3))

        return x, attn_weights_e, attn_weights_i
    
