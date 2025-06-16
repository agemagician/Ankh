from transformers import (
    T5EncoderModel,
    T5ForConditionalGeneration,
    AutoTokenizer,
    TFT5EncoderModel,
    TFT5ForConditionalGeneration,
    T5Tokenizer,
)
from enum import Enum
from typing import List, Tuple, Union
from ankh.models.utils import check_deprecated_args


class AvailableModels(Enum):
    """Ankh pre-trained model paths."""

    ANKH_BASE = "ElnaggarLab/ankh-base"
    ANKH_LARGE = "ElnaggarLab/ankh-large"
    ANKH3_LARGE = "ElnaggarLab/ankh3-large"
    ANKH3_XL = "ElnaggarLab/ankh3-xl"


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


def get_supported_frameworks() -> List:
    """Returns a `list` of the current supported model formats.

    Returns:
        List: Supported model formats.
    """
    return ["tf", "pt"]


def get_specified_model(
    path: str,
    generation: bool,
    output_attentions: bool,
    framework: str,
) -> Union[T5EncoderModel, T5ForConditionalGeneration]:
    if framework == "tf":
        return load_tf_model(path, generation, output_attentions)
    elif framework == "pt":
        return load_pt_model(path, generation, output_attentions)
    else:
        supported_frameworks = get_supported_frameworks()
        raise ValueError(
            "Expected framework to be one "
            f"of {supported_frameworks}. Received: {framework}"
        )


@check_deprecated_args(
    deprecated_args=["model_format"], new_args=["framework"]
)
def load_ankh_base(
    generation: bool = False,
    output_attentions: bool = False,
    framework: str = "pt",
) -> Tuple[T5EncoderModel, AutoTokenizer]:
    """Downloads and returns the base model and its tokenizer

    Args:
        generation (bool, optional): Whether to return
                                     `T5ForConditionalGeneration` will be
                                     returned otherwise `T5EncoderModel` will
                                     be returned. Defaults to False.
        output_attentions (bool, optional): Whether to return the attention or
                                            not. Defaults to False.

        framework (str, optional): The model format, currently supports
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
        framework=framework,
    )
    return model, tokenizer


@check_deprecated_args(
    deprecated_args=["model_format"], new_args=["framework"]
)
def load_ankh_large(
    generation: bool = False,
    output_attentions: bool = False,
    framework="pt",
) -> Tuple[Union[T5EncoderModel, T5ForConditionalGeneration], AutoTokenizer]:
    """Downloads and returns the large model and its tokenizer

    Args:
        generation (bool, optional): Whether to return
                                     `T5ForConditionalGeneration` will be
                                     returned otherwise `T5EncoderModel` will
                                     be returned. Defaults to False.
        output_attentions (bool, optional): Whether to return the attention or
                                            not. Defaults to False.
        framework (str, optional): The model format, currently supports
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
        framework=framework,
    )
    return model, tokenizer


@check_deprecated_args(
    deprecated_args=["model_format"], new_args=["framework"]
)
def load_ankh3_large(
    generation: bool = False,
    output_attentions: bool = False,
    framework="pt",
) -> Tuple[Union[T5EncoderModel, T5ForConditionalGeneration], AutoTokenizer]:
    """Downloads and returns the ankh3 large model and its tokenizer"""
    tokenizer = T5Tokenizer.from_pretrained(AvailableModels.ANKH3_LARGE.value)
    model = get_specified_model(
        path=AvailableModels.ANKH3_LARGE.value,
        generation=generation,
        output_attentions=output_attentions,
        framework=framework,
    )
    return model, tokenizer


@check_deprecated_args(
    deprecated_args=["model_format"], new_args=["framework"]
)
def load_ankh3_xl(
    generation: bool = False,
    output_attentions: bool = False,
    framework="pt",
) -> Tuple[Union[T5EncoderModel, T5ForConditionalGeneration], T5Tokenizer]:
    """Downloads and returns the ankh3 xl model and its tokenizer"""
    tokenizer = T5Tokenizer.from_pretrained(AvailableModels.ANKH3_XL.value)
    model = get_specified_model(
        path=AvailableModels.ANKH3_XL.value,
        generation=generation,
        output_attentions=output_attentions,
        framework=framework,
    )
    return model, tokenizer


@check_deprecated_args(
    deprecated_args=["model_format"], new_args=["framework"]
)
def load_model(
    model_name: str,
    generation: bool = False,
    output_attentions: bool = False,
    framework="pt",
) -> Tuple[
    Union[T5EncoderModel, T5ForConditionalGeneration],
    Union[AutoTokenizer, T5Tokenizer],
]:
    """Downloads and returns the specified model and its tokenizer

    Args:
        model_name (str): The name of the model to load.
        generation (bool, optional): Whether to return
                                     `T5ForConditionalGeneration` will be
                                     returned otherwise `T5EncoderModel` will
                                     be returned. Defaults to False.
        output_attentions (bool, optional): Whether to return the attention or
                                            not. Defaults to False.
        framework (str, optional): The model format, currently supports
                                    'tf' and 'pt'. Defaults to 'pt'.

    Returns:
        Tuple[Union[T5EncoderModel, T5ForConditionalGeneration],
        AutoTokenizer]: Returns T5 Model and its tokenizer.
    """
    available_models = {
        "ankh_base": load_ankh_base,
        "ankh_large": load_ankh_large,
        "ankh3_large": load_ankh3_large,
        "ankh3_xl": load_ankh3_xl,
    }
    if model_name in available_models:
        return available_models[model_name](
            generation=generation,
            output_attentions=output_attentions,
            framework=framework,
        )
    else:
        available_models = get_available_models()
        raise ValueError(
            f"Expected model_name to be one of {available_models}. "
            f"Received: {model_name}"
        )
