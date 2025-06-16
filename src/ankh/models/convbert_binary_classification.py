from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from ankh.models import layers


class ConvBERTForBinaryClassification(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = "max",
    ):
        super().__init__()

        """
            ConvBERT model for binary classification task.

            Args:
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
                pooling: String specifying the global pooling function.
                Accepts "avg" or "max". Default: "max"
        """
        if pooling is None:
            raise ValueError(
                "`pooling` cannot be `None` in a binary classification task. "
                "Expected ['avg', 'max']."
            )
        self.convbert = layers.ConvBERT(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
        )
        self.decoder = nn.Linear(input_dim, 1)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert(embed)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss, logits=logits, hidden_states=None, attentions=None
        )
