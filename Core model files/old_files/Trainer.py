# %%
# from Pipeline import *
from Pipeline import *
import time
import numpy as np
from evaluateur import *

from greedy_beam_search import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)


def auto_encoding_train(model, train_data, image_bool):
    if image_bool:
        data, features = train_data
        features = features.to(device=device, dtype=torch.float32)
    else:
        data, target = train_data
    if image_bool:
        output = model(data, True, features)

    else:
        output = model(data)
    # loss = model.criterion(output.mT,target)
    loss = model.criterion(
        output.mT,
        torch.cat(
            (
                data[:, 1:],
                torch.ones(data.shape[0], 1, dtype=torch.int)
                .fill_(model.padding_id)
                .to(device=device, dtype=torch.int),
            ),
            dim=1,
        ).to(device=device, dtype=data.dtype),
    )
    model.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    model.optimizer.step()
    return loss.item()


def cycle_consistent_forward(
    model_A, model_B, text_input, target, image_input=None, image_bool=False
):
    src_mask = model_A.generate_square_subsequent_mask(
        model_A.n_head * text_input.shape[0], text_input.shape[1]
    )  # square mask
    tgt_mask = model_A.generate_square_subsequent_mask(
        model_A.n_head * text_input.shape[0], text_input.shape[1]
    )
    src_padding_mask = (text_input == model_A.padding_id).to(device=device)
    tgt_padding_mask = (target == model_A.padding_id).to(device=device)
    memory_mask = model_A.generate_square_subsequent_mask(
        text_input.shape[0], text_input.shape[1]
    )
    memory_key_padding_mask = (text_input == model_A.padding_id).to(device=device)
    if image_bool:
        mem_ei_mask = torch.zeros(
            [
                text_input.shape[0],
                text_input.shape[1],
                text_input.shape[1] + image_input.shape[1],
            ]
        ).to(device=device, dtype=bool)
        mem_ei_mask[
            :, 0 : text_input.shape[1], 0 : text_input.shape[1]
        ] = model_A.generate_square_subsequent_mask(
            text_input.shape[0], text_input.shape[1]
        ).to(
            device=device
        )
        mem_ei_key_padding_mask = (text_input == model_A.padding_id).to(device=device)
        mem_ei_key_padding_mask = torch.cat(
            (
                mem_ei_key_padding_mask,
                torch.full([text_input.shape[0], image_input.shape[1]], False).to(
                    device=device
                ),
            ),
            dim=1,
        )
    memory = model_A.encoder(
        model_A.positional_encoder(model_A.embedding(text_input)),
        src_mask,
        src_padding_mask,
    )
    if not model_A.teacher_forcing:
        if np.random.rand() < 1:  # DO NOT TEACHER FORCE
            target = torch.cat(
                (
                    torch.ones(text_input.shape[0], 1, dtype=torch.int).fill_(
                        model_B.begin_id
                    ),
                    torch.ones(
                        text_input.shape[0], text_input.shape[1] - 1, dtype=torch.int
                    ).fill_(model_B.padding_id),
                ),
                dim=1,
            ).to(device)
    # target = torch.cat((target[:,1:],torch.ones(target.shape[0] ,1,dtype = torch.int).fill_(model_B.padding_id).to(device=device,dtype = torch.int)),dim =1).to(device=device,dtype = torch.int)
    if image_bool:
        mem_masks = [memory_mask, mem_ei_mask]
        mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
        image_encoded = model_A.feedforward(image_input)
        decoder_input = [
            model_B.positional_encoder(model_B.embedding(target)),
            image_encoded,
        ]
        output = model_B.decoder(
            decoder_input,
            memory,
            tgt_mask,
            mem_masks,
            tgt_padding_mask,
            mem_padding_masks,
            True,
        )[0]
    else:
        output = model_B.decoder(
            model_B.positional_encoder(model_B.embedding(target)),
            memory,
            tgt_mask,
            [memory_mask],
            tgt_padding_mask,
            [memory_key_padding_mask],
        )[0]
    return model_B.output_layer(output)


def cycle_consistency_train(model_A, model_B, train_data, image_bool=False):
    if image_bool:
        data, features = train_data
        features = features.to(device=device, dtype=torch.float32)
    else:
        data, target = train_data
    if image_bool:
        with torch.no_grad():
            first_output = torch.argmax(
                CCF_greedy(model_A, model_B, data, features, image_bool), dim=2
            )
            first_output = range_le_padding(
                check_data(
                    first_output, model_B.padding_id, model_B.begin_id, model_B.end_id
                ),
                model_B.padding_id,
            )
        output = cycle_consistent_forward(
            model_B, model_A, first_output, data, features, image_bool
        )
    else:
        with torch.no_grad():
            first_output = torch.argmax(CCF_greedy(model_A, model_B, data), dim=2)

            first_output = range_le_padding(
                check_data(
                    first_output, model_B.padding_id, model_B.begin_id, model_B.end_id
                ),
                model_B.padding_id,
            )
        output = cycle_consistent_forward(model_B, model_A, first_output, data)

    loss_A = model_A.criterion(
        output.mT,
        torch.cat(
            (
                data[:, 1:],
                torch.ones(data.shape[0], 1, dtype=torch.int)
                .fill_(model_A.padding_id)
                .to(device=device, dtype=torch.int),
            ),
            dim=1,
        ).to(device=device, dtype=data.dtype),
    )
    model_A.optimizer.zero_grad()
    model_B.optimizer.zero_grad()
    loss_A.backward()
    torch.nn.utils.clip_grad_norm_(model_A.parameters(), 5)
    torch.nn.utils.clip_grad_norm_(model_B.parameters(), 5)
    model_A.optimizer.step()
    model_B.optimizer.step()
    return loss_A.item()


import matplotlib.pyplot as plt
from livelossplot import PlotLosses


def mixed_train(
    val_data_en,
    val_data_fr,
    inv_map_en,
    inv_map_fr,
    model_fr,
    model_en,
    train_data_fr,
    train_data_en,
    n_iter,
    batch_size,
    image_bool=False,
    part_auto_encoding=1 / 2,
):
    loss_list = []
    liveloss = PlotLosses()
    bleu, meteor = evaluation(
        "greedy",
        val_data_en,
        val_data_fr,
        batch_size,
        model_en,
        model_fr,
        inv_map_en,
        inv_map_fr,
        image_bool,
    )
    model_fr.train()
    model_en.train()
    total_loss = 0
    start_time = time.time()
    tokenized_val_en = val_data_en[0]
    tokenized_val_fr = val_data_fr[0]
    for i_iter in range(n_iter):
        save_dataframe_eval(
            model_fr,
            model_en,
            val_data_en,
            val_data_fr,
            inv_map_en,
            inv_map_fr,
            image_bool,
            batch_size,
            i_iter,
        )
        model_en.train()
        model_fr.train()
        model_en.curr_epoch += 1
        model_fr.curr_epoch += 1
        batched_data_en, batched_data_fr = batchify(
            [train_data_en, train_data_fr], batch_size, image_bool
        )
        if image_bool:
            N = len(batched_data_fr[0])
        else:
            N = len(batched_data_fr)
        log_interval = N // 10
        for i in range(N):
            model_en.scheduler.step()
            model_fr.scheduler.step()
            U = np.random.rand()
            V = np.random.rand()
            if U < 1 / 2:  # ENGLISH DATA
                if image_bool:
                    train_data = get_batch(batched_data_en, i, image_bool)
                else:
                    train_data = get_batch(batched_data_en, i)
                model_A = model_en
                model_B = model_fr
            else:  # FRENCH DATA
                if image_bool:
                    train_data = get_batch(batched_data_fr, i, image_bool)
                else:
                    train_data = get_batch(batched_data_fr, i)
                model_A = model_fr
                model_B = model_en
            if V < part_auto_encoding:  # AUTO ENCODING
                loss = auto_encoding_train(model_A, train_data, image_bool)
                model_A.loss_list.append(loss)
            else:  # CYCLE CONSISTENT
                # print(train_data[0].shape)
                loss = cycle_consistency_train(model_A, model_B, train_data, image_bool)
                model_A.loss_list.append(loss)
                model_B.loss_list.append(loss)
            loss_list.append(loss)
            total_loss += loss
            if (i % log_interval == 0 and i != 0) or i == N - 1:
                with open(model_A.prefix + "logs.txt", "a") as logs:
                    logs.write(
                        "\nIteration : "
                        + str(i_iter)
                        + " batch numéro : "
                        + str(i)
                        + " en "
                        + str(
                            int(
                                1000
                                * (time.time() - start_time)
                                / (log_interval * batch_size)
                            )
                        )
                        + " ms par phrase, moyenne loss "
                        + str(total_loss / log_interval)
                        + " current lr "
                        + str(model_fr.scheduler.get_last_lr())
                        + " "
                        + str(model_en.scheduler.get_last_lr())
                    )
                    logs.close()
                bleu, meteor = evaluation(
                    "greedy",
                    val_data_en,
                    val_data_fr,
                    batch_size,
                    model_en,
                    model_fr,
                    inv_map_en,
                    inv_map_fr,
                    image_bool,
                )
                liveloss.update(
                    {
                        "Model FR mean training loss": np.mean(
                            model_fr.loss_list[-log_interval:]
                        ),
                        "Model EN mean training loss": np.mean(
                            model_fr.loss_list[-log_interval:]
                        ),
                        "BLEU score": bleu,
                        "METEOR score": meteor,
                        "EN LR": model_en.scheduler.get_last_lr()[0],
                        "FR LR": model_fr.scheduler.get_last_lr()[0],
                    }
                )
                liveloss.send()
                model_en.train()
                model_fr.train()
                total_loss = 0
                start_time = time.time()


# def differentiable_cycle_forward(model_A,model_B,text_input, image_input = None, image_bool = False, mask_ei = False):
#     src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask
#     tgt_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1])
#     src_padding_mask  = (text_input==  model_A.padding_id).to(device=device)
#     tgt_padding_mask = (text_input==  model_A.padding_id).to(device=device)
#     memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
#     memory_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
#     if image_bool and mask_ei:
#         mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device)
#         mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
#         mem_ei_key_padding_mask = (text_input ==  model_A.padding_id).to(device=device)
#         mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
#     else:
#         mem_ei_mask = None
#         mem_ei_key_padding_mask = None
#     text_encoded = model_A.encoder(model_A.positional_encoder(model_A.embedding(text_input)),src_mask,src_padding_mask)
#     if image_bool:
#         mem_masks = [memory_mask, mem_ei_mask]
#         mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
#         image_encoded = model_A.feedforward(image_input)
#         x = [text_encoded, image_encoded]
#         output = model_B.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
#     else:
#         x = text_encoded
#         output = model_B.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])
#     #Intermediate result to have the new masks
#     with torch.no_grad():
#         text_input = torch.argmax(model_B.output_layer(output),dim = 2)
#     #Compute new masks with augmented data
#     src_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1]) # square mask
#     tgt_mask = model_A.generate_square_subsequent_mask(model_A.n_head*text_input.shape[0],text_input.shape[1])
#     src_padding_mask  = (text_input==  model_B.padding_id).to(device=device)
#     tgt_padding_mask = (text_input== model_B.padding_id).to(device=device)
#     memory_mask = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1])
#     memory_key_padding_mask = (text_input == model_B.padding_id).to(device=device)
#     if image_bool:
#         mem_ei_mask = torch.zeros([text_input.shape[0], text_input.shape[1], text_input.shape[1] + image_input.shape[1]]).to(device=device)
#         mem_ei_mask[:,0:text_input.shape[1], 0:text_input.shape[1]] = model_A.generate_square_subsequent_mask(text_input.shape[0],text_input.shape[1]).to(device=device)
#         mem_ei_key_padding_mask = (text_input == model_B.padding_id).to(device=device)
#         mem_ei_key_padding_mask = torch.cat((mem_ei_key_padding_mask, torch.full([text_input.shape[0], image_input.shape[1]], False).to(device=device)), dim=1)
#     text_encoded = model_B.encoder(model_B.positional_encoder(output),src_mask,src_padding_mask)
#     if image_bool:
#         mem_masks = [memory_mask, mem_ei_mask]
#         mem_padding_masks = [memory_key_padding_mask, mem_ei_key_padding_mask]
#         image_encoded = model_A.feedforward(image_input)
#         x = [text_encoded, image_encoded]
#         output = model_A.decoder(x,model_B.positional_encoder(model_B.embedding(text_input)), tgt_mask , mem_masks , tgt_padding_mask, mem_padding_masks)
#     else:
#         x = text_encoded
#         output = model_A.decoder(x,model_A.positional_encoder(model_A.embedding(text_input)), tgt_mask , [memory_mask] , tgt_padding_mask, [memory_key_padding_mask])
#     return model_A.output_layer(output)

# def differentiable_cycle_consistency_train(model_A, model_B,train_data,image_bool=False):
#         if image_bool :
#             data, features = train_data
#         else :
#             data,target = train_data
#         if image_bool :
#             output = differentiable_cycle_forward(model_A,model_B, data, features, image_bool)
#             loss = model_A.criterion(output.mT,data)
#         else :
#             output = differentiable_cycle_forward(model_A,model_B, data)
#             loss = model_A.criterion(output.mT,target)
#         model_A.optimizer.zero_grad()
#         model_B.optimizer.zero_grad()
#         loss.backward()
#         torch.nn.utils.clip_grad_norm_(model_A.parameters(), 0.5)
#         torch.nn.utils.clip_grad_norm_(model_B.parameters(), 0.5)
#         model_A.optimizer.step()
#         model_B.optimizer.step()
#         return loss.item()
