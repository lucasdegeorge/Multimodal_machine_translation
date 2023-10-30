#%% 
import torch.nn as nn
import torch
from Decoder_Layers import *

with open("parameters.json", 'r') as f:
    parameters = json.load(f)
    device = parameters["device"]


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer() for i in range(parameters["nb_layers_dec"])])
        self.norm = nn.LayerNorm(parameters["d_model"])

    def forward(self, x, e_output, tgt_mask, tgt_key_padding_mask, e_key_padding_mask):

        for i in range(parameters["nb_layers_dec"]):
            x, attn_weights = self.layers[i](x, e_output, tgt_mask, tgt_key_padding_mask, e_key_padding_mask)
            
            if i==0:    # voir s'il est pas possible de faire quelque chose de plus propre
                mean_attn_weights = torch.sum(attn_weights, dim=1)
            else:
                mean_attn_weights += torch.sum(attn_weights, dim=1)

        x = self.norm(x)
        mean_attn_weights /= (parameters["nb_layers_dec"] * parameters["n_heads"])

        return x, mean_attn_weights
    

class Multimodal_Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Multimodal_DecoderLayer() for i in range(parameters["nb_layers_dec"])])
        self.norm = nn.LayerNorm(parameters["d_model"])

    def forward(self, x, e_output, i_output, tgt_mask, tgt_key_padding_mask, e_key_padding_mask, ei_key_padding_mask):

        for i in range(parameters["nb_layers_dec"]):
            x, attn_weights_e, attn_weights_i = self.layers[i](x, e_output, i_output, tgt_mask, tgt_key_padding_mask, e_key_padding_mask, ei_key_padding_mask)
            
            if i==0:    # voir s'il est pas possible de faire quelque chose de plus propre
                mean_attn_weights_e = torch.sum(attn_weights_e, dim=1)
                mean_attn_weights_i = torch.sum(attn_weights_i, dim=1)
            else:
                mean_attn_weights_e += torch.sum(attn_weights_e, dim=1)
                mean_attn_weights_i += torch.sum(attn_weights_i, dim=1)

        x = self.norm(x)
        mean_attn_weights_e /= (parameters["nb_layers_dec"] * parameters["n_heads"])
        mean_attn_weights_i /= (parameters["nb_layers_dec"] * parameters["n_heads"])

        return x, mean_attn_weights_e, mean_attn_weights_i
