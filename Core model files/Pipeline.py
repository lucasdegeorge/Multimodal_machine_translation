#%%Batchifier
import torch
from torch import Tensor
from typing import Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vocab() : 
    fichier_vocab_fr = open('vocab.fr')
    fichier_vocab_en = open('vocab.en')
    vocab_en = [line.split()[0] for line in fichier_vocab_en if len(line.split()) == 2]
    vocab_en = dict((y,x) for (x,y) in enumerate(vocab_en))
    vocab_fr = [line.split()[0] for line in fichier_vocab_fr if len(line.split()) == 2]
    vocab_fr = dict((y,x) for (x,y) in enumerate(vocab_fr))
    fichier_vocab_en.close()
    fichier_vocab_fr.close()
    return vocab_en, vocab_fr


def get_train_data_nouveau(batch_size, train_bool):

    vocab_en,vocab_fr = get_vocab()

    if train_bool : 
        fichier_train_fr = open('train.BPE.fr')
        fichier_train_en = open('train.BPE.en')
    else: 
        fichier_train_fr = open('val.BPE.fr')
        fichier_train_en = open('val.BPE.en')

    train_data_fr = [["DEBUT_DE_PHRASE"]+ligne.strip().split(" ")+['FIN_DE_PHRASE'] for ligne in fichier_train_fr ]
    train_data_en = [["DEBUT_DE_PHRASE"]+ligne.strip().split(" ")+["FIN_DE_PHRASE"] for ligne in fichier_train_en ]
    fichier_train_en.close()
    fichier_train_fr.close()
    # longueur_max = max(max([len(x) for x in train_data_fr]),max( [len(x) for x in train_data_fr]))
    longueur_max = 64

    train_data_fr = [[phrase[i] if i < len(phrase) else "TOKEN_VIDE" for i in range (longueur_max)] for phrase in train_data_fr]
    train_data_en = [[phrase[i] if i < len(phrase) else "TOKEN_VIDE" for i in range (longueur_max)] for phrase in train_data_en]

    #ATTENTION : DEMANDER IUN AVIS POUR CES DEUX BOUCLES QUI AJOUTENT AU VOCAB CE QUI MANQUE, CEST A DIRE '&@@' et ';@@' a cause des caracteres html
#     for ligne in train_data_en: 
#         for mot in ligne : 
#             if mot not in vocab_en: 
#                 # print(mot)
#                 vocab_en[mot] = len(vocab_en.keys())

#     for ligne in train_data_fr: 
#         for mot in ligne: 
#             if mot not in vocab_fr: 
#                 # print(mot)
#                 vocab_fr[mot] = len(vocab_fr.keys())
    for token in ['TOKEN_INCONNU','TOKEN_VIDE','DEBUT_DE_PHRASE','FIN_DE_PHRASE']:
        vocab_fr[token] = len(vocab_fr.keys())
        vocab_en[token] = len(vocab_en.keys())
    for i in range(len(train_data_en)):
        for j in range(len(train_data_en[i])):
            if train_data_en[i][j] not in vocab_en :
                train_data_en[i][j]="TOKEN_INCONNU"
    for i in range(len(train_data_fr)):
        for j in range(len(train_data_fr[i])):
            if train_data_fr[i][j]  not in vocab_fr : 
                train_data_fr[i][j]="TOKEN_INCONNU"

    # tokenized_fr = torch.zeros( len(train_data_fr),longueur_max)
    # tokenized_en = torch.zeros( len(train_data_en),longueur_max)
    # for i in range(len(train_data_fr)) : 
    #     for j in range (longueur_max) : 
    #         tokenized_fr[i,j] =  vocab_fr[train_data_fr[i][j]]
    #         tokenized_en[i,j] =  vocab_en[train_data_en[i][j]]

    # batched_fr = torch.tensor([[[vocab_fr[train_data_fr[k*batch_size+j][i]] for i in range (longueur_max)] for j in range(batch_size)] for k in range(len(train_data_fr)//batch_size)]).to(device=device, dtype= torch.long)
    # batched_en = torch.tensor([[[vocab_en[train_data_en[k*batch_size+j][i]] for i in range (longueur_max)] for j in range(batch_size)] for k in range(len(train_data_en)//batch_size)]).to(device=device, dtype= torch.long)
    tokenized_fr = torch.tensor([[vocab_fr[mot] for mot in phrase] for phrase in train_data_fr]).to(device = device, dtype = torch.long)
    tokenized_en = torch.tensor([[vocab_en[mot] for mot in phrase] for phrase in train_data_en]).to(device = device, dtype = torch.long)
    #En sortie : les données tokenizées non batchées
    return [tokenized_fr,tokenized_en, vocab_fr,vocab_en]

def batchify(data: Tensor, bsz: int, image_bool,conservative = False, permute= True) -> Tensor:
    """Divides the data into bsz separate sequences, removing extra elements
    that wouldn't cleanly fit.

    Args:
        data: Tensor, shape [N]
        bsz: int, batch size

    Returns:
        Tensor of shape [N // bsz, bsz]
    """
    if image_bool : 
        data_en,data_fr = data
        text_en,features = data_en
        text_fr,features = data_fr
        if permute : 
            permutation = torch.randperm(text_en.shape[0])
            text_en = text_en[permutation]
            text_fr = text_fr[permutation]
            features = features[permutation]
        if text_en.shape[0]%bsz != 0 and not conservative:
            text_en = text_en[:(text_en.shape[0]-text_en.shape[0]%bsz)]
            text_fr = text_fr[:(text_fr.shape[0]-text_fr.shape[0]%bsz)]
            features = features[:(text_fr.shape[0]-text_fr.shape[0]%bsz)] 
        if text_en.shape[0]%bsz != 0 and  conservative:
            text_en = torch.cat((text_en,torch.zeros(bsz-text_en.shape[0]%bsz,text_en.shape[1]).to(device = device, dtype = torch.long)),dim = 0)
            text_fr= torch.cat((text_fr,torch.zeros(bsz-text_fr.shape[0]%bsz,text_fr.shape[1]).to(device = device, dtype = torch.long)),dim = 0)
            features= torch.cat((features.to(device = device),torch.zeros(bsz-features.shape[0]%bsz,features.shape[1],features.shape[2],features.shape[2]).to(device = device, dtype = torch.float)),dim = 0)
        # print(text.shape)
        return [text_en.view(text_en.shape[0]//bsz, bsz, text_en.shape[1]), features.view(features.shape[0]//bsz, bsz , features.shape[1],features.shape[2]**2).transpose(-1,-2)],[text_fr.view(text_fr.shape[0]//bsz, bsz, text_fr.shape[1]), features.view(features.shape[0]//bsz, bsz , features.shape[1],features.shape[2]**2).transpose(-1,-2)]
    else : 
        text_en,text_fr = data
        if permute :
            permutation = torch.randperm(text_fr.shape[0])
            text_en = text_en[permutation]
            text_fr = text_fr[permutation]
        if text_en.shape[0]%bsz != 0 and not conservative:
            text_fr = text_fr[:(text_fr.shape[0]-text_fr.shape[0]%bsz)]
            text_en = text_en[:(text_en.shape[0]-text_en.shape[0]%bsz)]
        if text_en.shape[0]%bsz != 0 and  conservative:
            text_en = torch.cat((text_en,torch.zeros(bsz-text_en.shape[0]%bsz,text_en.shape[1]).to(device = device, dtype = torch.long)),dim = 0)
            text_fr= torch.cat((text_fr,torch.zeros(bsz-text_fr.shape[0]%bsz,text_fr.shape[1]).to(device = device, dtype = torch.long)),dim = 0)
            



        return text_en.view(text_en.shape[0]//bsz, bsz, text_en.shape[1]),text_fr.view(text_fr.shape[0]//bsz, bsz, text_fr.shape[1])
        
    
def get_batch(source,i, image_bool = False) : 
    if image_bool : 
        return source[0][i],source[1][i].to(device) #, dtype = torch.float16)
    else :
        return source[i],source[i]

# def get_batch(source: Tensor, i: int,device) -> Tuple[Tensor, Tensor]:
#     """
#     Args:
#         source: Tensor, shape [full_seq_len, batch_size]
#         i: int

#     Returns:
#         tuple (data, target), where data has shape [seq_len, batch_size] and
#         target has shape [seq_len * batch_size]
#     """
#     seq_len = min(bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i:i+seq_len].reshape(-1)

#     return data.to(device), target.to(device)

def check_data(data,padding_id,begin_id,end_id):
    for i in range(len(data)):
        if torch.equal(data[i] , padding_id*torch.ones_like(data[i])):
            data[i][0] = begin_id
            data[i][1] = end_id
    return data

def range_le_padding(data,padding_id) : 
    for i in range(data.size(0)):
        j=0
        while j < data.size(1): 
            if torch.equal(data[i][j:] , padding_id * torch.ones_like(data[i][j:])):
                break
            elif data[i][j] == padding_id : 
                data[i] = torch.cat((data[i][:j],data[i][j+1:],torch.Tensor([padding_id]).to(device= device, dtype = torch.int64)))
            else : 
                j+=1
    return data