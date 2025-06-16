from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import TokenClassifierOutput
from ankh.models import layers


class ConvBERTForMultiLabelClassification(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        """
            ConvBERT model for multilabel classification task.
            Args:
                num_tokens: Integer specifying the number of tokens that
                should be the output of the final layer.
                input_dim: Dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the
                `ConvBERT` model.
                hidden_dim: Integer specifying the hidden dimension for the
                `ConvBERT` model.
                num_hidden_layers: Integer specifying the number of hidden
                layers for the `ConvBERT` model.
                kernel_size: Integer specifying the filter size for the
                `ConvBERT` model. Default: 7
                dropout: Float specifying the dropout rate for the
                `ConvBERT` model. Default: 0.2
        """
        self.convbert = layers.ConvBERT(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=None,
        )
        self.num_labels = num_tokens
        self.decoder = nn.Linear(input_dim, num_tokens)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(
                logits.view(-1, self.num_labels),
                labels.view(-1, self.num_labels),
            )
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert(embed)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
