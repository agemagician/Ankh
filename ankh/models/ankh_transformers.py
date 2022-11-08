from transformers import T5EncoderModel, AutoTokenizer
from enum import Enum
import os
from typing import List, Tuple

# Huggingface pretrained models.
available_models = {
    "base_model": "ElnaggarLab/protx-base-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel768",
    "large_model": "ElnaggarLab/protx-large-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel1536",
}


class AvailableModels(Enum):
    BASE_MODEL = "ElnaggarLab/protx-base-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel768"
    LARGE_MODEL = "ElnaggarLab/protx-large-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel1536"


def get_available_models() -> List:
    """
    Returns a `list` of the current available pretrained models.
    Args:
        None
    
    Returns:
        List of available models.
    """
    return list(available_models.keys())


def load_base_model(
    output_attentions: bool = False,
) -> Tuple[T5EncoderModel, AutoTokenizer]:

    """
    Downloads and returns the base model and its tokenizer

    Args:
        output_attentions: Whether to return the attention tensors when making an inference. Default: False
    
    Returns:
        `T5EncoderModel` and `AutoTokenizer`
    """

    tokenizer = AutoTokenizer.from_pretrained(
        available_models["base_model"], use_auth_token=os.environ["huggingface_token"]
    )
    model = T5EncoderModel.from_pretrained(
        available_models["base_model"],
        use_auth_token=os.environ["huggingface_token"],
        output_attentions=output_attentions,
    )
    return model, tokenizer


def load_large_model(
    output_attentions: bool = False,
) -> Tuple[T5EncoderModel, AutoTokenizer]:

    """
    Downloads and returns the large model and its tokenizer

    Args:
        output_attentions: Whether to return the attention tensors when making an inference. Default: False
    
    Returns:
        `T5EncoderModel` and `AutoTokenizer`
    """

    tokenizer = AutoTokenizer.from_pretrained(
        available_models["base_model"], use_auth_token=os.environ["huggingface_token"]
    )
    model = T5EncoderModel.from_pretrained(
        available_models["large_model"],
        use_auth_token=os.environ["huggingface_token"],
        output_attentions=output_attentions,
    )
    return model, tokenizer


available_models_fns = {"base": load_base_model, "large": load_large_model}


def load_model(
    model_name: str, output_attentions: bool = False
) -> Tuple[T5EncoderModel, AutoTokenizer]:
    """
    Downloads and returns the specified model and its tokenizer

    Args:
        model_name: String specifying which model to load.
            - `base`: Returns the base model and its tokenizer.
            - `large`: Returns the large model and its tokenizer.
        output_attentions: Whether to return the attention tensors when making an inference. Default: False
    
    Returns:
        `T5EncoderModel` and `AutoTokenizer`
    """

    return available_models_fns[model_name](output_attentions=output_attentions)
