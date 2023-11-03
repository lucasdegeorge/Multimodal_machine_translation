import torch
import torch.nn as nn
import json
from .model import Encoder, Decoder, PositionalEncoder

with open("parameters.json", "r") as f:
    parameters = json.load(f)
    device = parameters["device"]


class Transformer(nn.module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(parameters["n_token"], parameters["d_model"])
        # self.trg_embedding = nn.Embedding(parameters["n_token"], parameters["d_model"])

        self.positional_encoder = PositionalEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(parameters["d_model"], self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        src_input,
        trg_input,
        tgt_attn_mask=None,
        tgt_padding_mask=None,
        src_padding_mask=None,
    ):
        """
        inputs:
            src_input: the source, the sentence to translate
            trg_input: the target, the sentence to translate to
            tgt_attn_mask: the mask for the decoder so that attention does not look at the future.
            tgt_padding_mask: the mask for the decoder so that attention does not look at the padding. We used padding to make sure all inputs have the same sequence lengths during batching.
            src_padding_mask: the mask for the encoder so that attention does not look at the padding. We used padding to make sure all inputs have the same sequence lengths during batching.
        outputs:
            output: the predicted sentence
            attn_weights_e: the attention weights of the text
        """
        src_input = self.embedding(src_input)  # (B, L) => (B, L, d_model)
        trg_input = self.embedding(trg_input)  # (B, L) => (B, L, d_model)
        src_input = self.positional_encoder(
            src_input
        )  # (B, L, d_model) => (B, L, d_model)
        trg_input = self.positional_encoder(
            trg_input
        )  # (B, L, d_model) => (B, L, d_model)

        e_output = self.encoder(src_input, src_padding_mask)  # (B, L, d_model)
        d_output, attn_weights_e = self.decoder(
            trg_input, e_output, tgt_attn_mask, tgt_padding_mask, src_padding_mask
        )  # (B, L, d_model)

        output = self.softmax(
            self.output_linear(d_output)
        )  # (B, L, d_model) => # (B, L, n_token)

        return output, attn_weights_e


class MultiModalTransformer(nn.module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(parameters["n_token"], parameters["d_model"])
        # self.trg_embedding = nn.Embedding(parameters["n_token"], parameters["d_model"])
        self.img_ff = nn.Linear(
            parameters["img_size"], parameters["d_model"]
        )  # selon ancien Modèle file ? j'en sais rien
        self.positional_encoder = PositionalEncoder()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.output_linear = nn.Linear(parameters["d_model"], self.trg_vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(
        self,
        src_input,
        trg_input,
        img_input,
        tgt_attn_mask=None,
        tgt_padding_mask=None,
        src_padding_mask=None,
        ei_padding_mask=None,
    ):
        """
        inputs:
            src_input: the source, the sentence to translate
            trg_input: the target, the sentence to translate to
            img_input: the image to "translate"
            tgt_attn_mask: the mask for the decoder so that attention does not look at the future.
            tgt_padding_mask: the mask for the decoder so that attention does not look at the padding. We used padding to make sure all inputs have the same sequence lengths during batching.
            src_padding_mask: the mask for the encoder so that attention does not look at the padding. We used padding to make sure all inputs have the same sequence lengths during batching.
            ei_padding_mask: ei is the concatenation of then encoded image and the encoded text. The encoded text was padded during batching so we need to mask the padding.
        outputs:
            output: the predicted sentence
            attn_weights_e: the attention weights of the text
            attn_weights_i: the attention weights of the image

        """
        src_input = self.embedding(src_input)
        trg_input = self.embedding(trg_input)

        src_input = self.positional_encoder(src_input)
        trg_input = self.positional_encoder(trg_input)

        i_output = self.img_ff(img_input)
        e_output = self.encoder(src_input, src_padding_mask)
        d_output, attn_weights_e, attn_weights_i = self.decoder(
            trg_input,
            e_output,
            i_output,
            tgt_attn_mask,
            tgt_padding_mask,
            src_padding_mask,
            ei_padding_mask,
        )

        output = self.softmax(self.output_linear(d_output))
        return output, attn_weights_e, attn_weights_i
