#%% 
import torch.nn as nn
import torch
from Decoder_Layers import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Decoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()

        self.layers = self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, tgt, memory, tgt_mask=None, e_mask=None, tgt_key_padding_mask=None, e_key_padding_mask=None):
        output = tgt
        for i, layer in enumerate(self.layers):
            output, attn_weights = layer(output, memory, tgt_mask=tgt_mask,
                                                                memory_mask=memory_mask,
                                                                tgt_key_padding_mask=tgt_key_padding_mask,
                                                            memory_key_padding_mask=memory_key_padding_mask,image_bool=image_bool)

            if i == 0:
                n_heads = attention_weights_e.shape[1]
                attention_weights_e_sum = attention_weights_e.sum(dim=1)
            
            else:
                attention_weights_e_sum += attention_weights_e.sum(dim=1)
            
        attention_weights_e_sum = attention_weights_e_sum/(self.num_layers*n_heads)
        if self.norm is not None:
            output = self.norm(output)
        return output,attention_weights_e_sum
        

class MultiModal_Decoder(nn.Module):
    
    __constants__ = ['norm']

    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__()
        #torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}") je savaias pas à quoi cette ligne servait
        self.layers = self.layers = nn.ModuleList([decoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        

    def forward(self, tgt, memory, tgt_mask = None,
                memory_mask = None, tgt_key_padding_mask = None,
                memory_key_padding_mask = None, image_bool=False):
       
        output = tgt
        if image_bool:
            for i,mod in enumerate(self.layers):

                output, attention_weights_e, attention_weights_i = mod(output, memory, tgt_mask=tgt_mask,
                                                                 memory_mask=memory_mask,
                                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                                memory_key_padding_mask=memory_key_padding_mask,image_bool=image_bool)
                
                output = output,tgt[1]
                if i == 0:
                    n_heads = attention_weights_i.shape[1]
                    attention_weights_e_sum = attention_weights_e.sum(dim=1)
                    attention_weights_i_sum = attention_weights_i.sum(dim=1)
                else:
                    attention_weights_e_sum += attention_weights_e.sum(dim=1)
                    attention_weights_i_sum += attention_weights_i.sum(dim=1)
            output  = output[0]
            attention_weights_e_sum = attention_weights_e_sum/(self.num_layers*n_heads)
            attention_weights_i_sum = attention_weights_i_sum/(self.num_layers*n_heads)
            if self.norm is not None:
                output = self.norm(output)
            return output,attention_weights_e_sum,attention_weights_i_sum
        else:
            for i,mod in enumerate(self.layers):
        
                output,attention_weights_e = mod(output, memory, tgt_mask=tgt_mask,
                                                                 memory_mask=memory_mask,
                                                                 tgt_key_padding_mask=tgt_key_padding_mask,
                                                                memory_key_padding_mask=memory_key_padding_mask,image_bool=image_bool)

                if i == 0:
                    n_heads = attention_weights_e.shape[1]
                    attention_weights_e_sum = attention_weights_e.sum(dim=1)
                
                else:
                    attention_weights_e_sum += attention_weights_e.sum(dim=1)
                
            attention_weights_e_sum = attention_weights_e_sum/(self.num_layers*n_heads)
            if self.norm is not None:
                output = self.norm(output)
            return output,attention_weights_e_sum