from transformers import T5EncoderModel, T5ForConditionalGeneration, AutoTokenizer
from enum import Enum
import os
from typing import List, Tuple


class AvailableModels(Enum):
    """
        Ankh pre-trained model paths.
    """

    ANKH_BASE = "ElnaggarLab/protx-base-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel768"
    ANKH_LARGE = "ElnaggarLab/protx-large-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel1536"


def get_available_models() -> List:
    """
        Returns a `list` of the current available pretrained models.

        Args:
            None

        Returns:
            List of available models.
    """
    return [o.name.lower() for o in AvailableModels]


def load_base_model(
    generation: bool = False,
    output_attentions: bool = False,
) -> Tuple[T5EncoderModel, AutoTokenizer]:

    """
        Downloads and returns the base model and its tokenizer

        Args:
            output_attentions: Whether to return the attention tensors when making an inference. Default: False

        Returns:
            `T5ForConditionalGeneration` if `generation=True` and `T5EncoderModel` otherwise
            `AutoTokenizer`
    """

    # Temporary until the pre-trained models become public.
    auth_token = os.environ.get("huggingface_token", None)

    if auth_token is None:
        raise ValueError(
            f"Currently, The pre-trained models are private. "
            f"Make sure that `huggingface_token` is set as a global environment variable. "
            f"This error should be removed when the pre-trained models become public."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        AvailableModels.ANKH_BASE.value, use_auth_token=auth_token
    )
    if generation:
        model = T5ForConditionalGeneration.from_pretrained(
            AvailableModels.ANKH_BASE.value,
            use_auth_token=auth_token,
            output_attentions=output_attentions,
        )
    else:
        model = T5EncoderModel.from_pretrained(
            AvailableModels.ANKH_BASE.value,
            use_auth_token=auth_token,
            output_attentions=output_attentions,
        )
    return model, tokenizer


def load_large_model(
    generation: bool = False,
    output_attentions: bool = False,
) -> Tuple[T5EncoderModel, AutoTokenizer]:

    """
        Downloads and returns the large model and its tokenizer

        Args:
            output_attentions: Whether to return the attention tensors when making an inference. Default: False

        Returns:
            `T5ForConditionalGeneration` if `generation=True` and `T5EncoderModel` otherwise
            `AutoTokenizer`
    """

    # Temporary until the pre-trained models become public.
    auth_token = os.environ.get("huggingface_token", None)

    if auth_token is None:
        raise ValueError(
            f"Currently, The pre-trained models are private. "
            f"Make sure that `huggingface_token` is set as a global environment variable. "
            f"This error should be removed when the pre-trained models become public."
        )

    tokenizer = AutoTokenizer.from_pretrained(
        AvailableModels.ANKH_LARGE.value, use_auth_token=auth_token
    )
    if generation:
        model = T5ForConditionalGeneration.from_pretrained(
            AvailableModels.ANKH_LARGE.value,
            use_auth_token=auth_token,
            output_attentions=output_attentions,
        )
    else:
        model = T5EncoderModel.from_pretrained(
            AvailableModels.ANKH_LARGE.value,
            use_auth_token=auth_token,
            output_attentions=output_attentions,
        )
    return model, tokenizer


available_models_fns = {"base": load_base_model, "large": load_large_model}


def load_model(
    model_name: str, generation: bool = False, output_attentions: bool = False
) -> Tuple[T5EncoderModel, AutoTokenizer]:
    """
        Downloads and returns the specified model and its tokenizer
    
        Args:
            model_name: String specifying which model to load.
                - `base`: Returns the base model and its tokenizer.
                - `large`: Returns the large model and its tokenizer.
            output_attentions: Whether to return the attention tensors when making an inference. Default: False
    
        Returns:
            `T5ForConditionalGeneration` if `generation=True` and `T5EncoderModel` otherwise
            `AutoTokenizer`
    """

    return available_models_fns[model_name](
        generation=generation, output_attentions=output_attentions
    )