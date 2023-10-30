import torchmetrics
from greedy_beam_search import *
from Pipeline import *
import numpy as np
from nltk.translate.meteor_score import meteor_score
def tensor_to_sentence(output,inv_dic):
    result = [inv_dic[int(x)] for x in output]
    sentence = ""
    for word in result : 
        if word == "DEBUT_DE_PHRASE" or word == "TOKEN_VIDE" :
            pass
        elif '@@' in word: 
            sentence+=word[:-2]
        elif word == "FIN_DE_PHRASE" :
            break 
        else :
            sentence+=word +" "
    return sentence

def cut_list_at_value(L,value):
    if value in L :
        index = L.index(value)
        return L[:index]
    else :
        return L
def give_tokens(output, padding_id, end_id ) : #takes output of greedy search tensor of size [bsz,seqlen,n_token]. returns the tokens, with no padding before end of sentence token
    #todoso, just need to take the 2nd outpuuts
    values, indices = torch.kthvalue(output, 2 , dim = 2)
    sentences = torch.argmax(output, dim  = 2 ) # size bsz, seqlen
    #we modify this sentences tensor
    for i in range(sentences.size(0)):#batch
        authorize_padding = False
        for j in range(sentences.size(1)):#sentence
            if sentences[i][j] == end_id:
                authorize_padding = True
            elif sentences[i][j] == padding_id and not authorize_padding : 
                sentences[i][j] = indices[i][j]
    return sentences


def traduit(mode,model_A,model_B,src, inv_map_src,image_bool,tgt,inv_map_tgt,j):
    model_A.eval()
    model_B.eval()
    if image_bool : 
        data,features= src
    else :
        data= src
    # 
    with open(model_A.prefix+"logs.txt",'a') as logs :
        if image_bool :
            if mode == 'greedy':
                output = give_tokens(CCF_greedy(model_A,model_B,data, features, True),model_B.padding_id,model_B.end_id)[j]
            else : 
                output = CCF_beam_search(model_A, model_B, data, 3, features, True)[j]
        else :
            if mode == 'greedy':
                output = give_tokens(CCF_greedy(model_A,model_B,data, None, False),model_B.padding_id,model_B.end_id)[j]
            else : 
                output = CCF_beam_search(model_A, model_B, data, 3, None, False)[j]
        logs.write("\nBleu score\n")
        bleu = evalue_bleu(output.view(-1),inv_map_tgt,tgt.view(-1))
        logs.write(str(bleu))
        logs.write("\nMeteor score\n")
        meteor = evalue_meteor(output.view(-1),inv_map_tgt,tgt.view(-1))
        logs.write(str(meteor))
        logs.write("\nBonne traduction\n")
        logs.write(str(tensor_to_sentence(tgt.view(-1),inv_map_tgt)))
        logs.write("\nOutput\n")
        logs.write(str(output))
        logs.write("\nAuto encoding\n")
        # print(tensor_to_sentence(torch.argmax(model_A(data,True,features),dim = 2).view(-1),inv_map_src))
        if image_bool :
            logs.write(str(tensor_to_sentence(torch.argmax(model_A(data,True,features),dim = 2)[j].view(-1),inv_map_src)))
        else :
            logs.write(str(tensor_to_sentence(torch.argmax(model_A(data,False),dim = 2)[j].view(-1),inv_map_src)))
        logs.close()
    return tensor_to_sentence(output.view(-1),inv_map_tgt),bleu,meteor

def evalue_bleu(output,inv_dic_tgt, tgt):
    sacre_bleu =torchmetrics.SacreBLEUScore()
    result = [inv_dic_tgt[int(x)] for x in output if inv_dic_tgt[int(x)] not in  ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
    target = [inv_dic_tgt[int(x)] for x in tgt if inv_dic_tgt[int(x)] not in  ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
    return sacre_bleu( result,[target])
def evalue_meteor(output,inv_dic_tgt, tgt):
    result = [inv_dic_tgt[int(x)] for x in output if inv_dic_tgt[int(x)] not in  ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
    target = [inv_dic_tgt[int(x)] for x in tgt if inv_dic_tgt[int(x)] not in  ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
    return meteor_score([target], result)

def donne_random(i,j,val_data_en,val_data_fr,batch_size,image_bool):
    batched_data_en,batched_data_fr=batchify([val_data_en,val_data_fr],batch_size,image_bool)
    if image_bool:
        src,features = batched_data_en
        tgt,_ = batched_data_fr
        return src[i],features[i],tgt[i]
    else :
        src,tgt = batched_data_en,batched_data_fr
        return src[i],tgt[i]

def evaluation(mode,val_data_en,val_data_fr,batch_size,model_en,model_fr,inv_map_en,inv_map_fr,image_bool):
    if image_bool :
        tokenized_val_en = val_data_en[0]
    else : 
        tokenized_val_en = val_data_en
    i = np.random.randint(len(tokenized_val_en)//batch_size)
    j = np.random.randint(batch_size)
    if image_bool: 
        src,features,tgt = donne_random(i,j,val_data_en,val_data_fr,batch_size,image_bool)
        features = features.to(device,dtype=torch.float32)
        data = [src,features]
    else :
        src,tgt = donne_random(i,j,val_data_en,val_data_fr,batch_size,image_bool)
        data = src
    
    trad,bleu,meteor = traduit(mode,model_en,model_fr,data, inv_map_en,image_bool,tgt[j],inv_map_fr,j) 
    with open(model_en.prefix+"logs.txt",'a') as logs :
        logs.write("\nPhrase à traduire : \n" + tensor_to_sentence(src[j],inv_map_en)+ "\nPhrase traduite : \n"+ trad)
        logs.close()
    return bleu, meteor

# import torchtext
import pandas as pd
def dataframe_eval(model_fr,model_en,val_data_en,val_data_fr,inv_map_en,inv_map_fr,image_bool,batch_size,mode= "greedy") :
    model_fr.eval()
    model_en.eval()
    batched_data_en,batched_data_fr=batchify([val_data_en,val_data_fr],batch_size,image_bool,conservative=True,permute = False)
    if image_bool:
        src,features = batched_data_en
        tgt,_ = batched_data_fr
    else :
        src,tgt = batched_data_en,batched_data_fr
    traductions_en_fr = []
    references_en = []
    traductions_fr_en = []
    references_fr = []
    traductions_en_fr_txt_only = []
    traductions_fr_en_txt_only = []
    for batch in range(len(src)):
        print(batch)
        if image_bool :
            src[batch],features[batch],tgt[batch] = src[batch].to(device),features[batch].to(device),tgt[batch].to(device)
            if mode == "greedy":
                traduction = torch.argmax(CCF_greedy(model_en,model_fr,src[batch],features[batch],image_bool) ,dim = 2)
            else : 
                traduction = CCF_beam_search(model_en, model_fr, src[batch], beam_size=3, image_input=features[batch], image_bool=True,get_attention=False)
            for i in range(traduction.shape[0]):
                temp = cut_list_at_value([inv_map_fr[traduction[i][j].item()]  for j in range(traduction.shape[1])],"FIN_DE_PHRASE")
                traductions_en_fr.append([x for x in temp if x not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
                references_fr.append([inv_map_fr[tgt[batch][i][j].item()] for j in range(tgt[batch].shape[1]) if inv_map_fr[tgt[batch][i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
            if mode =="greedy":
                traduction = torch.argmax(CCF_greedy(model_fr,model_en,tgt[batch],features[batch],image_bool) ,dim = 2)
            else : 
                traduction = CCF_beam_search(model_fr, model_en, tgt[batch], beam_size=3, image_input=features[batch], image_bool=True,get_attention=False)
            
            for i in range(traduction.shape[0]):
                temp = cut_list_at_value([inv_map_en[traduction[i][j].item()]  for j in range(traduction.shape[1])],"FIN_DE_PHRASE")
                traductions_fr_en.append([x for x in temp if x not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
                references_en.append([inv_map_en[src[batch][i][j].item()] for j in range(src[batch].shape[1]) if inv_map_en[src[batch][i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
            if mode =="greedy":
                traduction = torch.argmax(CCF_greedy(model_en,model_fr,src[batch],None,False) ,dim = 2)
            else : 
                traduction = CCF_beam_search(model_en, model_fr, src[batch], beam_size=3, image_input=None, image_bool=False,get_attention=False)
            
            for i in range(traduction.shape[0]):
                temp = cut_list_at_value([inv_map_fr[traduction[i][j].item()]  for j in range(traduction.shape[1])],"FIN_DE_PHRASE")
                traductions_en_fr_txt_only.append([x for x in temp if x not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
            if mode =="greedy":
                traduction = torch.argmax(CCF_greedy(model_fr,model_en,tgt[batch],None,False) ,dim = 2)
            else :
                traduction = CCF_beam_search(model_fr, model_en, tgt[batch], beam_size=3, image_input=None, image_bool=False,get_attention=False)
            
            for i in range(traduction.shape[0]):
                temp = cut_list_at_value([inv_map_en[traduction[i][j].item()]  for j in range(traduction.shape[1])],"FIN_DE_PHRASE")
                traductions_fr_en_txt_only.append([x for x in temp if x not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
        else :
            src[batch] ,tgt[batch]= src[batch].to(device),tgt[batch].to(device)
            if mode =="greedy": 
                traduction = torch.argmax(CCF_greedy(model_en,model_fr,src[batch],None,image_bool) ,dim = 2)
            else : 
                traduction = CCF_beam_search(model_en, model_fr, src[batch], beam_size=3, image_input=None, image_bool=False,get_attention=False)
            
            for i in range(traduction.shape[0]):
                temp = cut_list_at_value([inv_map_fr[traduction[i][j].item()]  for j in range(traduction.shape[1])],"FIN_DE_PHRASE")
                traductions_fr_en.append([x for x in temp if x not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
                references_fr.append([inv_map_fr[tgt[batch][i][j].item()] for j in range(tgt[batch].shape[1]) if inv_map_fr[tgt[batch][i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
            if mode =="greedy":
                traduction = torch.argmax(CCF_greedy(model_fr,model_en,tgt[batch],None,image_bool) ,dim = 2)
            else : 
                traduction = CCF_beam_search(model_fr, model_en, tgt[batch], beam_size=3, image_input=None, image_bool=False,get_attention=False)
            
            for i in range(traduction.shape[0]):
                temp = cut_list_at_value([inv_map_en[traduction[i][j].item()]  for j in range(traduction.shape[1])],"FIN_DE_PHRASE")
                traductions_en_fr.append([x for x in temp if x not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])
                references_en.append([inv_map_en[src[batch][i][j].item()] for j in range(src[batch].shape[1]) if inv_map_en[src[batch][i][j].item()] not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]])

    if image_bool:
        data = {"traductions_en_fr":traductions_en_fr,"references_fr":references_fr,"traductions_fr_en":traductions_fr_en,"references_en":references_en,"traductions_en_fr_txt_only":traductions_en_fr_txt_only,"traductions_fr_en_txt_only":traductions_fr_en_txt_only}
    else :
        data = {"traductions_en_fr":traductions_en_fr,"references_fr":references_fr,"traductions_fr_en":traductions_fr_en,"references_en":references_en}

    df = pd.DataFrame(data)
    return df.loc()[:val_data_fr[0].shape[0]-1]
def bleu(row,langue_src):
    sacre_bleu = torchmetrics.SacreBLEUScore()
    if langue_src == "en":
        candidates= list(row["traductions_en_fr"])
        references = [list(row["references_fr"])]
        return sacre_bleu(candidates,references)
    else : 
        candidates= list(row["traductions_fr_en"])
        references = [list(row["references_en"])]
        return sacre_bleu(candidates,references)
def bleu_txt_only(row,langue_src):
    sacre_bleu = torchmetrics.SacreBLEUScore()
    if langue_src == "en":
        candidates= list(row["traductions_en_fr_txt_only"])
        references = [list(row["references_fr"])]
        return sacre_bleu(candidates,references)
    else : 
        candidates= list(row["traductions_fr_en_txt_only"])
        references = [list(row["references_en"])]
        return sacre_bleu(candidates,references)
    
def save_dataframe_eval(model_fr,model_en,val_data_en,val_data_fr,inv_map_en,inv_map_fr,image_bool,batch_size,epoch = 0,mode = "greedy"):
    df = dataframe_eval(model_fr,model_en,val_data_en,val_data_fr,inv_map_en,inv_map_fr,image_bool,batch_size,mode)
    df["bleu_en_fr"] = df.apply(lambda row: bleu(row,"en"), axis=1)
    df["bleu_fr_en"] = df.apply(lambda row: bleu(row,"fr"), axis=1)
    if image_bool:
        df["bleu_en_fr_txt_only"] = df.apply(lambda row: bleu_txt_only(row,"en"), axis=1)
        df["bleu_fr_en_txt_only"] = df.apply(lambda row: bleu_txt_only(row,"fr"), axis=1)
    df.to_csv(str(epoch)+model_fr.prefix+"_eval.csv")
def repeater(tensor):
    tensor = tensor.cpu()
    tensor = tensor.detach().numpy()
    tensor = np.repeat(tensor,14,axis=0)
    tensor = np.repeat(tensor,14,axis=1)
    tensor = torch.tensor(tensor)
    return tensor
import matplotlib.pyplot as plt
def plot_attention_on_image(image,attention,titre):

    attention = repeater(attention) 
    image = image.cpu()
    image = image.detach().numpy()
    image = image.transpose(1,2,0)
    plt.imshow(image)
    plt.imshow(attention,alpha = 0.3,cmap = "inferno")
    plt.colorbar()
    plt.savefig("Graphs attention/"+titre +"attention_image.png")
    plt.show()
    
def plot_attention_text(phrase, attention_text,titre):
    fig, ax = plt.subplots(1,1)
    attention_text = attention_text[:len(phrase)][:len(phrase)]
    plt.imshow(attention_text,cmap = "inferno")
    ax.set_xticks([i for i in range(len(phrase))])
    ax.set_xticklabels(phrase,rotation = 90)
    ax.set_yticks([i for i in range(len(phrase))])
    ax.set_yticklabels(phrase)
    plt.colorbar()
    plt.savefig("Graphs attention/"+titre+"attention_text.png")
def graphiques_attention(model_fr,model_en,batched_data_en,inv_map_en,inv_map_fr,batch_size,images,titres) :
    model_fr.eval()
    model_en.eval()

    src,features = batched_data_en
    traductions_en_fr = []
    references_en = []

    for batch in range(len(src)):

        src[batch],features[batch] = src[batch].to(device),features[batch].to(device)
        traduction,attention_e,attention_i= CCF_greedy(model_en,model_fr,src[batch],features[batch],True)
        traduction = torch.argmax(traduction ,dim = 2)
        for i in range(traduction.shape[0]):
            #Attention text
            phrase  = cut_list_at_value([inv_map_en[src[batch][i][j].item()] for j in range(src[batch].shape[1])] ,"FIN_DE_PHRASE")
            phrase = [x for x in phrase if x not in ["TOKEN_VIDE","DEBUT_DE_PHRASE","FIN_DE_PHRASE"]]
            plot_attention_text(phrase,attention_e[i],titres[i])

            #Attention image
            plot_attention_on_image(images[i],attention_i[i],titres[i])
