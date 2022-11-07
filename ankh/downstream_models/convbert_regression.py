
import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
import transformers.models.convbert as c_bert
from functools import partial


class ConvBertForRegression(nn.Module):
    def __init__(self, ninp, nhead, nhid, nlayers, convsize=9, dropout=0.5, training_labels_mean=None):
        super(ConvBertForRegression, self).__init__()
        
        self.model_type = 'Transformer'
        self.training_labels_mean = training_labels_mean

        
        encoder_layers_Config = c_bert.ConvBertConfig(hidden_size=ninp,
                                   num_attention_heads=nhead,
                                   intermediate_size=nhid,
                                   conv_kernel_size=convsize,
                                   num_hidden_layers=nlayers,
                                   hidden_dropout_prob=dropout)
       
        self.transformer_encoder = c_bert.ConvBertLayer(encoder_layers_Config)
        
        self.global_max_pooling = partial(torch.max, dim=1)
        
        self.decoder = nn.Linear(ninp, 1)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.training_labels_mean is not None:
            self.decoder.bias.data.fill_(self.training_labels_mean)
        else:
            self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, embed, labels=None):
        hidden_inputs = self.transformer_encoder(embed)[0]
        hidden_inputs, _ = self.global_max_pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        
        if labels is not None:
            loss = F.mse_loss(logits, labels)
        else:
            loss = None
        
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )