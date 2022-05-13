 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pdb import set_trace as stop
from .transformer_layers import SelfAttnLayer
from .utils import custom_replace,weights_init
from .position_enc import PositionEmbeddingSine,positionalencoding2d

 
class TransfmorerModel(nn.Module):
    def __init__(self, num_labels=6, layers=6,heads=8,dropout=0.1):
        super(TransfmorerModel, self).__init__()
        hidden = 512 # this should match the backbone output feature size
        
        # Label Embeddings
        self.label_input = torch.Tensor(np.arange(num_labels)).view(1,-1).long() #Tensor(1,80)
        self.label_lt = torch.nn.Embedding(num_labels, hidden, padding_idx=None) #Embedding(80, 2048)

        # Transformer
        self.self_attn_layers = nn.ModuleList([SelfAttnLayer(hidden,heads,dropout) for _ in range(layers)])
        self.output_linear = torch.nn.Linear(hidden, num_labels) #Linear（2048，18）

        # Other
        self.LayerNorm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

        # Init all except pretrained backbone
        self.label_lt.apply(weights_init)
        # self.known_label_lt.apply(weights_init)
        self.LayerNorm.apply(weights_init)
        self.self_attn_layers.apply(weights_init)
        self.output_linear.apply(weights_init)

    def forward(self, images):
        const_label_input = self.label_input.repeat(images.size(0),1).cuda() #Tensor(batchsize, 80)
        init_label_embeddings = self.label_lt(const_label_input)#Tensor(batchsize, 80, hidden)

        embeddings = init_label_embeddings

        # Feed label embeddings through Transformer
        embeddings = self.LayerNorm(embeddings)        
        attns = []
        for layer in self.self_attn_layers:
            embeddings,attn = layer(embeddings,mask=None)
            attns += attn.detach().unsqueeze(0).data

        # Readout each label embedding using a linear layer
        label_embeddings = embeddings[:,-init_label_embeddings.size(1):,:]# Tensor(2, 80, hidden)
        output = self.output_linear(label_embeddings) # Tensor(8, 18, 18)
        diag_mask = torch.eye(output.size(1)).unsqueeze(0).repeat(output.size(0),1,1).cuda() #Tensor(2, 80, 80)
        output = (output*diag_mask).sum(-1)# Tensor(2, 80) attns:Tensor(2, 404, 404)

        return output, attns

