from transformers import (
    T5EncoderModel,
    T5ForConditionalGeneration,
    AutoTokenizer,
    TFT5EncoderModel,
    TFT5ForConditionalGeneration,
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


def load_tf_model(path, generation, output_attentions):
    if generation:
        return TFT5ForConditionalGeneration.from_pretrained(
            path, output_attentions=output_attentions, from_pt=True
        )
    else:
        return TFT5EncoderModel.from_pretrained(
            path, output_attentions=output_attentions, from_pt=True
        )


def load_pt_model(path, generation, output_attentions):
    if generation:
        return T5ForConditionalGeneration.from_pretrained(
            path, output_attentions=output_attentions
        )
    else:
        return T5EncoderModel.from_pretrained(
            path, output_attentions=output_attentions
        )


def load_onnx_model(path, generation, output_attentions):
    raise NotImplementedError('`load_onnx_model` is not implemented yet.')


class SupportedModelFormatFunctions(Enum):
    PT = load_pt_model
    TF = load_tf_model
    ONNX = load_onnx_model


def get_supported_model_format_functions() -> List:
    """Returns a `list` of the current supported model formats.

    Returns:
        List: Supported model formats.
    """
    return [o.name.lower() for o in SupportedModelFormatFunctions]


def get_specified_model(path: str,
                        generation: bool,
                        output_attentions: bool,
                        model_format: str):

    model_loading_function = getattr(SupportedModelFormatFunctions,
                                     model_format.upper())

    return model_loading_function(path,
                                  generation=generation,
                                  output_attentions=output_attentions)


def load_base_model(
    generation: bool = False,
    output_attentions: bool = False,
    model_format: str = 'pt',
) -> Tuple[T5EncoderModel, AutoTokenizer]:
    """Downloads and returns the base model and its tokenizer

    Args:
        generation (bool, optional): Whether to return
                                     `T5ForConditionalGeneration` will be
                                     returned otherwise `T5EncoderModel` will
                                     be returned. Defaults to False.
        output_attentions (bool, optional): Whether to return the attention or
                                            not. Defaults to False.

        model_format (str, optional): The model format, currently supports
                                      'tf' and 'pt'. Defaults to 'pt'.

    Returns:
        Tuple[Union[T5EncoderModel, T5ForConditionalGeneration],
        AutoTokenizer]: Returns T5 Model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(AvailableModels.ANKH_BASE.value)
    model = get_specified_model(
        path=AvailableModels.ANKH_BASE.value,
        generation=generation,
        output_attentions=output_attentions,
        model_format=model_format,
    )
    return model, tokenizer


def load_large_model(
    generation: bool = False,
    output_attentions: bool = False,
    model_format='pt',
) -> Tuple[Union[T5EncoderModel, T5ForConditionalGeneration], AutoTokenizer]:
    """Downloads and returns the large model and its tokenizer

    Args:
        generation (bool, optional): Whether to return
                                     `T5ForConditionalGeneration` will be
                                     returned otherwise `T5EncoderModel` will
                                     be returned. Defaults to False.
        output_attentions (bool, optional): Whether to return the attention or
                                            not. Defaults to False.
        model_format (str, optional): The model format, currently supports
                                      'tf' and 'pt'. Defaults to 'pt'.

    Returns:
        Tuple[Union[T5EncoderModel, T5ForConditionalGeneration],
        AutoTokenizer]: Returns T5 Model and its tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(AvailableModels.ANKH_LARGE.value)
    model = get_specified_model(
        path=AvailableModels.ANKH_LARGE.value,
        generation=generation,
        output_attentions=output_attentions,
        model_format=model_format,
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
