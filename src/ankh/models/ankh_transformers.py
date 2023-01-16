from transformers import (
    T5EncoderModel,
    T5ForConditionalGeneration,
    AutoTokenizer
)
from enum import Enum
from typing import List, Tuple, Union


class AvailableModels(Enum):
    """Ankh pre-trained model paths."""

    ANKH_BASE = "ElnaggarLab/ankh-base"
    ANKH_LARGE = "ElnaggarLab/ankh-large"


def get_available_models() -> List:
    """Returns a `list` of the current available pretrained models.

    Returns:
        List: Available models.
    """
    return [o.name.lower() for o in AvailableModels]


def load_base_model(
    generation: bool = False,
    output_attentions: bool = False,
) -> Tuple[T5EncoderModel, AutoTokenizer]:
    """Downloads and returns the base model and its tokenizer

    Args:
        generation (bool, optional): Whether to return
                                     `T5ForConditionalGeneration` will be
                                     returned otherwise `T5EncoderModel` will
                                     be returned. Defaults to False.
        output_attentions (bool, optional): Whether to return the attention or
                                            not. Defaults to False.

    Returns:
        Tuple[Union[T5EncoderModel, T5ForConditionalGeneration],
        AutoTokenizer]: Returns T5 Model and its tokenizer.
    """

    tokenizer = AutoTokenizer.from_pretrained(AvailableModels.ANKH_BASE.value)
    if generation:
        model = T5ForConditionalGeneration.from_pretrained(
            AvailableModels.ANKH_BASE.value,
            output_attentions=output_attentions,
        )
    else:
        model = T5EncoderModel.from_pretrained(
            AvailableModels.ANKH_BASE.value,
            output_attentions=output_attentions,
        )
    return model, tokenizer


def load_large_model(
    generation: bool = False,
    output_attentions: bool = False,
) -> Tuple[Union[T5EncoderModel, T5ForConditionalGeneration], AutoTokenizer]:
    """Downloads and returns the large model and its tokenizer

    Args:
        generation (bool, optional): Whether to return
                                     `T5ForConditionalGeneration` will be
                                     returned otherwise `T5EncoderModel` will
                                     be returned. Defaults to False.
        output_attentions (bool, optional): Whether to return the attention or
                                            not. Defaults to False.

    Returns:
        Tuple[Union[T5EncoderModel, T5ForConditionalGeneration],
        AutoTokenizer]: Returns T5 Model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(AvailableModels.ANKH_LARGE.value)
    if generation:
        model = T5ForConditionalGeneration.from_pretrained(
            AvailableModels.ANKH_LARGE.value,
            output_attentions=output_attentions,
        )
    else:
        model = T5EncoderModel.from_pretrained(
            AvailableModels.ANKH_LARGE.value,
            output_attentions=output_attentions,
        )
    return model, tokenizer


available_models_fns = {"base": load_base_model, "large": load_large_model}


def load_model(
    model_name: str, generation: bool = False, output_attentions: bool = False
) -> Tuple[Union[T5EncoderModel, T5ForConditionalGeneration], AutoTokenizer]:
    """Downloads and returns the specified model and its tokenizer

    Args:
        model_name (str): Model name, Expects "base" or "large"
        generation (bool, optional): Whether to return
                                     `T5ForConditionalGeneration` will be
                                     returned otherwise `T5EncoderModel` will
                                     be returned. Defaults to False.
        output_attentions (bool, optional): Whether to return the attention or
                                            not. Defaults to False.

    Returns:
        Tuple[Union[T5EncoderModel, T5ForConditionalGeneration],
        AutoTokenizer]: Returns T5 Model and its tokenizer.
    """

    return available_models_fns[model_name](
        generation=generation, output_attentions=output_attentions
    )
