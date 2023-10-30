#%%
import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import json

with open("parameters.json", 'r') as f:
    parameters = json.load(f)
    device = parameters["device"]   


class MultiModalAttention(nn.Module):
    def __init__(self):
        super().__init__()

        d_model = parameters["d_model"]

        self.d_model = d_model
        self.d_k = d_model // parameters["n_heads"]
        self.h = parameters["n_heads"]

        self.lambda1 = parameters["lambda1"]
        self.lambda2 = parameters["lambda2"]

        self.q_linear = nn.Linear(d_model,d_model)
        # For text (indice e) 
        self.v_e_linear = nn.Linear(d_model, d_model)
        self.k_e_linear = nn.Linear(d_model, d_model)
        # For image (indice i)
        self.v_i_linear = nn.Linear(d_model,d_model)
        self.k_i_linear = nn.Linear(d_model,d_model)
        # For both text and image (indice ei)
        self.v_ei_linear = nn.Linear(d_model,d_model)
        self.k_ei_linear = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(parameters["dropout"]) 
        self.out = nn.Linear(d_model, d_model)


    def forward(self, q, k_e, k_i, k_ei, v_e, v_i, v_ei, mask_e=None, mask_ei=None, e_padding_mask=None, ei_padding_mask=None):
        bs = q.size(1)
        q = self.q_linear(q).view(-1, q.size(1), self.h, self.d_k) 
        q = q.transpose(1,2)
        
        # Score for text
        k_e = self.k_e_linear(k_e).view(-1, k_e.size(1), self.h, self.d_k) 
        k_e = k_e.transpose(1,2)
        v_e = self.v_e_linear(v_e).view(-1, v_e.size(1), self.h, self.d_k)
        v_e = v_e.transpose(1,2)
        
        scores_e, attn_weights_e = self.attention(q, k_e, v_e, self.d_k, mask_e, e_padding_mask, self.dropout, only_image=False)

        # Score for image : 
        k_i = self.k_i_linear(k_i).view(-1, k_i.size(1), self.h, self.d_k) 
        k_i = k_i.transpose(1,2)
        v_i = self.v_i_linear(v_i).view(-1, v_i.size(1), self.h, self.d_k)
        v_i = v_i.transpose(1,2)
        scores_i, attn_weights_i = self.attention(q, k_i, v_i, self.d_k, None, None, self.dropout, only_image=True)

        # Score for text and image : 
        k_ei = self.k_ei_linear(k_ei).view(-1, k_ei.size(1), self.h, self.d_k) 
        k_ei = k_ei.transpose(1,2)
        v_ei = self.v_ei_linear(v_ei).view(-1, v_ei.size(1), self.h, self.d_k)
        v_ei = v_ei.transpose(1,2)
        scores_ei, attn_weights_ei = self.attention(q, k_ei, v_ei, self.d_k, mask_ei, ei_padding_mask, self.dropout, only_image=False)

        # final scores 
        scores = scores_e + self.lambda1 * scores_i + self.lambda2 * scores_ei
        
        concat = scores.transpose(1,2).contiguous().view(-1, bs, self.d_model)
        output = self.out(concat)
        
        return output, attn_weights_e, attn_weights_i 
    

    def attention(self, q, k, v, d_k, mask=None, padding_mask=None, dropout=None, only_image=False):
        output = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)
        if not(only_image):
            if mask is not None:
                mask = mask.unsqueeze(1)
                output = output.masked_fill(mask == True, float('-inf'))
            if padding_mask is not None:
                output = output.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        
        attn_weights = F.softmax(output, dim=-1)
        if dropout is not None:
            output = dropout(attn_weights)

        output = torch.matmul(attn_weights, v)
        return output, attn_weights

        

