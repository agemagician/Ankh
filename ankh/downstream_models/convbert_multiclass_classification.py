import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import TokenClassifierOutput
import transformers.models.convbert as c_bert


class ConvBertForMultiClassClassification(nn.Module):
    def __init__(self, num_tokens, input_dim, nhead, hidden_dim, nlayers, convsize=7, dropout=0.2):
        super(ConvBertForMultiClassClassification, self).__init__()
        '''
            ConvBert model for binary classification task.
            Args:
                num_tokens: Integer specifying the number of tokens that should be the output of the final layer.
                input_dim: the dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the `ConvBert` model.
                hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
                nlayers: Integer specifying the number of layers for the `ConvBert` model.
                convsize: Integer specifying the filter size for the `ConvBert` model. Default: 7
                dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
        '''

        self.model_type = "Transformer"
        self.num_labels = num_tokens

        encoder_layers_Config = c_bert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=convsize,
            num_hidden_layers=nlayers,
            hidden_dropout_prob=dropout,
        )

        self.transformer_encoder = c_bert.ConvBertLayer(encoder_layers_Config)
        self.decoder = nn.Linear(input_dim, num_tokens)

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, embd, labels=None):
        hidden_inputs = self.transformer_encoder(embd)[0]
        logits = self.decoder(hidden_inputs)

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
