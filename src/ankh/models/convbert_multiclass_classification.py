from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import TokenClassifierOutput
from ankh.models import layers


class ConvBertForMultiClassClassification(layers.BaseModule):
    def __init__(
        self,
        num_tokens: int,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super(ConvBertForMultiClassClassification, self).__init__(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=None,
        )
        """
            ConvBert model for multiclass classification task.

            Args:
                num_tokens: Integer specifying the number of tokens that should be the output of the final layer.
                input_dim: Dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the `ConvBert` model.
                hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
                num_hidden_layers: Integer specifying the number of hidden layers for the `ConvBert` model.
                num_layers: Integer specifying the number of `ConvBert` layers.
                kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
                dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
        """

        self.model_type = "Transformer"
        self.num_labels = num_tokens
        self.decoder = nn.Linear(input_dim, num_tokens)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.num_labels), labels.view(-1))
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
