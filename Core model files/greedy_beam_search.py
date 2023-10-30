#%% Librairies 

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import xavier_uniform_
import heapq
import copy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 256

#%% Greedy 
 
def CCF_greedy(model_A,model_B,text_input, image_input = None, image_bool = False,get_attention = False): 
    max_len = 64

    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask 
    src_padding_mask  = (text_input== model_A.padding_id).to(device=device)
    
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)

    if image_bool:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
    if image_bool:
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)

    decoder_input = torch.cat((torch.ones(batch_size, 1, dtype = torch.int).fill_(model_B.begin_id),torch.ones(batch_size ,max_len-1,dtype = torch.int).fill_(model_B.padding_id)),dim =1)

    for i in range(max_len-1):

        tgt_mask = model_B.generate_square_subsequent_mask(model_B.n_head*decoder_input.shape[0],decoder_input.shape[1])
        tgt_padding_mask = (decoder_input ==  model_B.padding_id).to(device=device)
        memory_mask = model_A.generate_square_subsequent_mask(decoder_input.shape[0],decoder_input.shape[1])
        memory_key_padding_mask = (decoder_input ==  model_B.padding_id).to(device=device)
        
        if image_bool:
            mem_ei_mask = torch.zeros([decoder_input.shape[0], decoder_input.shape[1], decoder_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
            mem_ei_mask[:,0:decoder_input.shape[1], 0:decoder_input.shape[1]] = model_A.generate_square_subsequent_mask(decoder_input.shape[0],decoder_input.shape[1]).to(device=device)
            mem_ei_key_padding_mask = (decoder_input ==  model_B.padding_id).to(device=device)
            mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([decoder_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)

        if image_bool :  
            x = [model_B.positional_encoder(model_B.embedding(decoder_input.to(device))), image_encoded]
            if not get_attention: 
                output = model_B.decoder(x,text_encoded, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks,image_bool)[0]
            else :
                output,attention_e,attention_i = model_B.decoder(x,text_encoded, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks,image_bool)
            
        else:
            x = text_encoded
            if not get_attention:
                output = model_B.decoder(model_B.positional_encoder(model_B.embedding(decoder_input.to(device))),x, tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask],image_bool)[0]
            else :
                output,attention_e = model_B.decoder(x,text_encoded, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks,image_bool)
            
        # Greedy 
        prob =  model_B.output_layer(output)
        next_words = torch.argmax(prob, dim=2)[:,i+1]
        decoder_input[:,i+1] = next_words
    if not get_attention:
        return model_B.output_layer(output)
    else :
        if image_bool:
            return model_B.output_layer(output),attention_e,attention_i
        else : 
            return model_B.output_layer(output),attention_e

#%% Beam search 

def CCF_beam_search(model_A, model_B, text_input, beam_size=3, image_input=None, image_bool=False,get_attention=False):
    max_len = 64
    device = text_input.device

    src_mask = model_A.generate_square_subsequent_mask(model_A.n_head * text_input.shape[0], text_input.shape[1])
    src_padding_mask = (text_input == model_A.padding_id).to(device=device)

    # Encode the text input
    text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)), src_mask, src_padding_mask)

    if image_bool:
        mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
        mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
        mem_ei_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
    memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
    memory_key_padding_mask = (text_input ==  model_B.padding_id).to(device=device)
    if image_bool:
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)

    # Initialize the beam
    beam = [ [ torch.cat((torch.ones(batch_size, 1, dtype = torch.int).fill_(model_B.begin_id),torch.ones(batch_size ,max_len-1,dtype = torch.int).fill_(model_B.padding_id)),dim =1), torch.zeros(batch_size)] ]

    # Loop until the maximum length is reached
    for i in range(max_len - 1):
        # print("i="+str(i))
        new_beam = []
        for seq, seq_score in beam:   # dim of seq : batch_size * i 
            # Get the last token of the sequence
            last_token = seq

            # If the last token is the end-of-sequence token, add the sequence to the new beam
            # if last_token.item() == model_B.end_id:
            #     new_beam.append((seq, seq_score))
            #     continue

            tgt_mask = model_B.generate_square_subsequent_mask(model_B.n_head*last_token.shape[0],last_token.shape[1])
            tgt_padding_mask = (last_token ==  model_B.padding_id).to(device=device)
            memory_mask = model_A.generate_square_subsequent_mask(last_token.shape[0],last_token.shape[1])
            memory_key_padding_mask = (last_token ==  model_B.padding_id).to(device=device)
            if image_bool:
                mem_ei_mask = torch.zeros([last_token.shape[0], last_token.shape[1], last_token.shape[1] + image_input.shape[1]]).to(device=device,dtype = bool)
                mem_ei_mask[:,0:last_token.shape[1], 0:last_token.shape[1]] = model_A.generate_square_subsequent_mask(last_token.shape[0],last_token.shape[1]).to(device=device)
                mem_ei_key_padding_mask = (last_token ==  model_B.padding_id).to(device=device)
                mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([last_token.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
            
            # Decode the next token
            if image_bool:
                x = [model_B.positional_encoder(model_B.embedding(last_token.to(device))), image_encoded]
                if not get_attention: 
                    output = model_B.decoder(x,text_encoded, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks,image_bool)[0]
                else :
                    output,attention_e,attention_i = model_B.decoder(x,text_encoded, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks,image_bool)
            else:
                x = text_encoded
                if not get_attention:
                    output = model_B.decoder(model_B.positional_encoder(model_B.embedding(last_token.to(device))),x, tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask],image_bool)[0]
                else :
                    output,attention_e = model_B.decoder(x,text_encoded, tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks,image_bool)
            
            # Get the top k candidates using log probabilities
            # log_probs = torch.log_softmax(model_B.output_layer(output)[:, -1, :], dim=-1) LUCAS VERSION
            log_probs = torch.sum(torch.log_softmax(model_B.output_layer(output),dim = 2)[:,:i+1,:] , dim = 1)
            top_k_scores, top_k_tokens = torch.topk(log_probs, k=beam_size, dim=-1,largest = True) # Dim : [batch_size, beam_size]

            # Add the top k candidates to the new beam
            for j in range(beam_size):
                new_seq = copy.deepcopy(seq)
                new_seq[:,i+1] = top_k_tokens[:, j].view(-1,1).squeeze()
                # new_seq = torch.cat([seq, top_k_tokens[:, j].unsqueeze(1)], dim=-1) 
                # new_score = seq_score.to(device) - top_k_scores[:, j].to(device) VERSION LUCAS
                new_score = top_k_scores[:, j].to(device)
                new_beam.append((new_seq, new_score))

            # beam = [ [torch.cat((torch.ones(batch_size, i+2, dtype = torch.int).fill_(model_B.begin_id),torch.ones(batch_size,97-(i+2),dtype = torch.int).fill_(model_B.padding_id)),dim =1) , torch.zeros(16)] for _ in range(beam_size) ]
            beam = [ [torch.ones(batch_size, max_len,dtype = torch.int).fill_(model_B.padding_id) , torch.zeros(batch_size)] for _ in range(beam_size) ]
#            
            for k in range(batch_size):
                scores_path = torch.zeros(len(new_beam))
                for path in range(len(new_beam)):
                    # get the tokens/scores for one sentence
                    scores_path[path] = new_beam[path][1][k].item()
                _, indices = torch.topk(scores_path, beam_size,largest = True)
                for id in range(len(indices)):
                    beam[id][0][k] = new_beam[indices[id]][0][k]
                    beam[id][1][k] = scores_path[id]

    # Return the top sequence
    if not get_attention:
        return beam[0][0]
    else :
        if image_bool:
            return beam[0][0],attention_e,attention_i
        else : 
            return beam[0][0],attention_e
 

#%% Data for tests : 

# from Modele_decodeur_maison import Modèle
# from Pipeline import get_train_data_nouveau, batchify

# def get_batch(source,i, image_bool = False): 
#     if image_bool : 
#         return source[0][i],source[1][i].to(device, dtype = torch.float32)
#     else :
#         return source[i],source[i]

# # Texts
# tokenized_fr,tokenized_en, vocab_fr,vocab_en = get_train_data_nouveau(batch_size)
# #Data non batchés
# n_token_fr = len(vocab_fr.keys())
# n_token_en = len(vocab_en.keys())

# n_head =4 
# num_encoder_layers = 4
# num_decoder_layers = 4
# dim_feedforward = 1024
# dropout = 0.1
# activation = nn.Softmax(dim=2)
# embedding_dim = 512

# model_fr = Modèle(n_token_fr,embedding_dim,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation,vocab_fr["TOKEN_VIDE"],vocab_fr["DEBUT_DE_PHRASE"],vocab_fr["FIN_DE_PHRASE"]).to(device)
# model_en = Modèle(n_token_en,embedding_dim,n_head, num_encoder_layers,num_decoder_layers,dim_feedforward,dropout,activation,vocab_en["TOKEN_VIDE"],vocab_en["DEBUT_DE_PHRASE"],vocab_en["FIN_DE_PHRASE"]).to(device)

# # With images
# # train_features = np.load("C:/Users/lucas/Desktop/train-resnet50-res4frelu.npy")
# # train_features = torch.from_numpy(train_features)
# # train_data_en = [tokenized_en, train_features]
# # batched_data = batchify(train_data_en,batch_size,True)
# # texts, images = get_batch(batched_data, 0, True)

# # Text only 
# # train_data_en = tokenized_en
# # batched_data = batchify(train_data_en,batch_size,False)
# # data = batched_data

# #%% Tests 

# # A = CCF_beam_search(model_fr,model_en, texts,image_input=images, image_bool=True)
# B = CCF_beam_search(model_fr,model_en, texts)
