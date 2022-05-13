import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stop
from .utils import get_activation_fn
from torch.nn import TransformerDecoder as TransformerDecoder

def padding_mask(seq_k, seq_q):
    # seq_k 和 seq_q 的形状都是 [B,L]
    len_q = seq_q.size(1)
    # `PAD` is 0
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)  # shape [B, L_q, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                    diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # [B, L, L]
    return mask

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        # Custom method to return attn outputs. Otherwise same as nn.TransformerEncoderLayer
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)

    def forward(self, src, src_mask= None, src_key_padding_mask = None):
        src2,attn = self.self_attn(src, src, src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2) 
        src = self.norm1(src)
        src = self.linear1(src)
        src = self.activation(src)
        src = self.dropout(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.linear2(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src,attn

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = nn.ReLU(True)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt2 = self.self_attn(tgt, tgt, tgt)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.multihead_attn(tgt, memory, memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

class SelfAttnLayer(nn.Module):
    def __init__(self, d_model, nhead = 4,dropout=0.1):
        super().__init__()
        self.transformer_layer_enc = TransformerEncoderLayer(d_model, nhead, d_model*1, dropout=dropout, activation='relu')
        self.transformer_layer_dec = TransformerDecoderLayer(d_model=512, nhead=8)
        # self.transformer_layer_dec = TransformerDecoderLayer(d_model, nhead, d_model*1, dropout=dropout, activation='relu')
        # self.transformer_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model, dropout=dropout, activation='gelu')

    def forward(self,k,mask=None):
        attn = None
        k=k.transpose(0,1)  
        enc_x, enc_attn = self.transformer_layer_enc(k, src_mask=mask)
        # dec_self_attn_pad_mask = padding_mask(k, k)
        dec_x, dec_attn = self.transformer_layer_dec(k, enc_x)
        # x = self.transformer_layer(k,src_mask=mask)
        # dec_x, dec_attn = self.transformer_layer_dec(k)
        # x=enc_x.transpose(0,1)
        x = dec_x.transpose(0,1)
        return x