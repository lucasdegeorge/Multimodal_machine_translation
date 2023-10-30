#%% 
import torch.nn as nn
import torch
from Multimodal_Attention import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DecoderLayer(nn.Module):   # for text only, we use the classical attention layer
    def __init__(self, d_model, heads, dim_feedforward, dropout=0.1):
        super().__init__()

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model, heads, dropout, batch_first=True)   
        self.attn_2 = nn.MultiheadAttention(d_model, heads, dropout, batch_first=True)
        
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
    def __init__(self, d_model, heads, dim_feedforward, dropout=0.1, is_there_image=True):
        super().__init__()
        self.is_there_image = is_there_image

        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = nn.MultiheadAttention(d_model, heads, dropout, batch_first=True)
        self.attn_2 = MultiModalAttention(d_model, heads, dropout, lambda1=1, lambda2=1)

        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model)
            )  
        
    def forward(self, x, e_output, i_output, tgt_mask, tgt_key_padding_mask, e_key_padding_mask, ei_key_padding_mask):
        # x as the same role as x in the classical transformer decoder layer
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
    
