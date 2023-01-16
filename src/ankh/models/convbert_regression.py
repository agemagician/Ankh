from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import SequenceClassifierOutput
from ankh.models import layers


class ConvBertForRegression(layers.BaseModule):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int,
        num_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = "max",
        training_labels_mean: float = None,
    ):
        if pooling is None:
            raise ValueError(
                '`pooling` cannot be `None` in a regression task. Expected ["avg", "max"].'
            )

        super(ConvBertForRegression, self).__init__(
            input_dim=input_dim,
            nhead=nhead,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            pooling=pooling,
        )
        """
            ConvBert model for regression task.

            Args:
                input_dim: Dimension of the input embeddings.
                nhead: Integer specifying the number of heads for the `ConvBert` model.
                hidden_dim: Integer specifying the hidden dimension for the `ConvBert` model.
                num_hidden_layers: Integer specifying the number of hidden layers for the `ConvBert` model.
                num_layers: Integer specifying the number of `ConvBert` layers.
                kernel_size: Integer specifying the filter size for the `ConvBert` model. Default: 7
                dropout: Float specifying the dropout rate for the `ConvBert` model. Default: 0.2
                pooling: String specifying the global pooling function. Accepts "avg" or "max". Default: "max"
                training_labels_mean: Float specifying the average of the training labels. Useful for faster and better training. Default: None
        """

        self.training_labels_mean = training_labels_mean
        self.decoder = nn.Linear(input_dim, 1)
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        if self.training_labels_mean is not None:
            self.decoder.bias.data.fill_(self.training_labels_mean)
        else:
            self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def _compute_loss(self, logits, labels):
        if labels is not None:
            loss = F.mse_loss(logits, labels)
        else:
            loss = None
        return loss

    def forward(self, embed, labels=None):
        hidden_inputs = self.convbert_forward(embed)
        hidden_inputs = self.pooling(hidden_inputs)
        logits = self.decoder(hidden_inputs)
        loss = self._compute_loss(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
