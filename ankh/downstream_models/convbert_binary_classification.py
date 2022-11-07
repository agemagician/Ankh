import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
import transformers.models.convbert as c_bert
from functools import partial


class ConvBertForBinaryClassification(nn.Module):
    def __init__(self, input_dim, nhead, hidden_dim, nlayers, convsize=7, dropout=0.2):
        super(ConvBertForBinaryClassification, self).__init__()

        """
            ConvBert model for binary classification task.
            Args:
                input_dim: the dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the `ConvBert` model.
                hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
                nlayers: Integer specifying the number of layers for the `ConvBert` model.
                convsize: Integer specifying the filter size for the `ConvBert` model. Default: 7
                dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
        """

        self.model_type = "Transformer"

        encoder_layers_Config = c_bert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=convsize,
            num_hidden_layers=nlayers,
            hidden_dropout_prob=dropout,
        )

        self.transformer_encoder = c_bert.ConvBertLayer(encoder_layers_Config)

        # Max pooling over the timesteps.
        self.global_max_pooling = partial(torch.max, dim=1)

        self.decoder = nn.Linear(input_dim, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, embed, labels=None):
        hidden_inputs = self.transformer_encoder(embed)[0]
        hidden_inputs, _ = self.global_max_pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = None

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=None, attentions=None
        )
