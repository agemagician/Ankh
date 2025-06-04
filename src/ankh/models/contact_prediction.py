import torch
import torch.nn as nn
import torch.nn.functional as F
from ankh.models.layers import ConvBERT
from typing import Optional


class ContactPredictionHead(nn.Module):
    def __init__(self, input_dim: int, num_tokens: int = 2):
        super().__init__()
        self.decoder = nn.Linear(input_dim * 2, num_tokens)
        self.reset_parameters()

    def reset_parameters(self):
        initrange = 0.1
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def compute_loss(
        self,
        logits: torch.FloatTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        if labels is not None:
            # Calculate per-element losses
            per_element_loss = F.cross_entropy(
                logits.view(-1, self.decoder.out_features),
                labels.view(-1),
                ignore_index=-1,
                reduction="none",
            )

            # Create a mask for valid (non-ignored) labels
            valid_mask = (labels.view(-1) != -1)
            num_valid_labels = valid_mask.sum()

            # Sum losses only for valid labels. If no valid labels, sum is 0.0.
            sum_of_valid_losses = per_element_loss[valid_mask].sum()

            # Clamp the number of valid labels to a minimum of 1.0
            # for the division to prevent division by zero.
            # If num_valid_labels is 0, sum_of_valid_losses is also 0,
            # so loss becomes 0.0 / 1.0 = 0.0.
            divisor = torch.clamp(num_valid_labels.float(), min=1.0)

            loss = sum_of_valid_losses / divisor
            return loss
        else:
            return None

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        labels: torch.LongTensor | None = None,
    ) -> torch.FloatTensor:
        prod = hidden_states[:, :, None, :] * hidden_states[:, None, :, :]
        diff = hidden_states[:, :, None, :] - hidden_states[:, None, :, :]
        pairwise_features = torch.cat((prod, diff), -1)
        logits = self.decoder(pairwise_features)
        logits = (logits + logits.transpose(1, 2)) / 2

        loss = self.compute_loss(logits, labels)

        return {
            "loss": loss,
            "logits": logits,
        }


class ConvBERTForContactPrediction(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_hidden_layers: int = 1,
        kernel_size: int = 7,
        dropout: float = 0.1,
        num_tokens: int = 2,
    ):
        super().__init__()

        self.encoder = ConvBERT(
            input_dim=input_dim,
            nhead=num_heads,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.decoder = ContactPredictionHead(input_dim, num_tokens)

    def get_extended_attention_mask(
        self, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        extended_attention_mask = attention_mask[:, None, None, :]
        extended_attention_mask = (
            1.0 - extended_attention_mask
        ) * torch.finfo(attention_mask.dtype).min
        return extended_attention_mask

    def forward(
        self,
        embd: torch.FloatTensor,
        attention_mask: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
    ):
        if attention_mask is not None:
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask.to(embd.dtype)
            )
        else:
            extended_attention_mask = None

        hidden_states = self.encoder(
            embd, attention_mask=extended_attention_mask
        )[0]

        output = self.decoder(hidden_states, labels)

        return output
