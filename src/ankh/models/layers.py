from torch import nn
from functools import partial
import torch
from transformers.models import convbert
from typing import Optional


class GlobalMaxPooling1D(nn.Module):
    def __init__(self):
        """
        Applies global max pooling over timesteps dimension
        """

        super().__init__()
        self.global_max_pool1d = partial(torch.max, dim=1)

    def forward(self, x):
        out, _ = self.global_max_pool1d(x)
        return out


class GlobalAvgPooling1D(nn.Module):
    def __init__(self):
        """
        Applies global average pooling over timesteps dimension
        """

        super().__init__()
        self.global_avg_pool1d = partial(torch.mean, dim=1)

    def forward(self, x):
        out = self.global_avg_pool1d(x)
        return out


class ConvBERT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        nhead: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.2,
        pooling: str = None,
    ):
        """
        Base model that consists of ConvBert layer.

        Args:
            input_dim: Dimension of the input embeddings.
            nhead: Integer specifying the number of heads for the
            `ConvBert` model.
            hidden_dim: Integer specifying the hidden dimension for the
            `ConvBert` model.
            nlayers: Integer specifying the number of layers for the
            `ConvBert` model.
            kernel_size: Integer specifying the filter size for the
            `ConvBert` model. Default: 7
            dropout: Float specifying the dropout rate for the
            `ConvBert` model. Default: 0.2
            pooling: String specifying the global pooling function.
            Accepts "avg" or "max". Default: "max"
        """
        super().__init__()

        self.model_type = "Transformer"
        encoder_layers_Config = convbert.ConvBertConfig(
            hidden_size=input_dim,
            num_attention_heads=nhead,
            intermediate_size=hidden_dim,
            conv_kernel_size=kernel_size,
            hidden_dropout_prob=dropout,
            num_hidden_layers=num_hidden_layers,
        )

        self.encoder = convbert.ConvBertModel(encoder_layers_Config).encoder

        if pooling is not None:
            if pooling in {"avg", "mean"}:
                self.pooling = GlobalAvgPooling1D()
            elif pooling == "max":
                self.pooling = GlobalMaxPooling1D()
            else:
                raise ValueError(
                    "Expected pooling to be [`avg`, `max`]. "
                    f"Received: {pooling}"
                )

    def get_extended_attention_mask(
        self,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (
            1.0 - extended_attention_mask
        ) * torch.finfo(attention_mask.dtype).min
        return extended_attention_mask

    def forward(
        self,
        x: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        attention_mask = self.get_extended_attention_mask(attention_mask)
        x = self.encoder(x, attention_mask=attention_mask)[0]
        x = self.pooling(x)
        return x
