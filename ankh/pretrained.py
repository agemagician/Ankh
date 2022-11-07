from transformers import T5EncoderModel, AutoTokenizer
from enum import Enum
import os
from typing import List, Tuple


available_models = {
    "base_model": "ElnaggarLab/protx-base-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel768",
    "large_model": "ElnaggarLab/protx-large-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel1536",
}


class AvailableModels(Enum):
    BASE_MODEL = "ElnaggarLab/protx-base-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel768"
    LARGE_MODEL = "ElnaggarLab/protx-large-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel1536"


def get_available_models() -> List:
    return list(available_models.keys())


def load_base_model(
    output_attentions: bool = False,
) -> Tuple[T5EncoderModel, AutoTokenizer]:
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
    tokenizer = AutoTokenizer.from_pretrained(
        available_models["base_model"], use_auth_token=os.environ["huggingface_token"]
    )
    model = T5EncoderModel.from_pretrained(
        available_models["large_model"],
        use_auth_token=os.environ["huggingface_token"],
        output_attentions=output_attentions,
    )
    return model, tokenizer


available_models_fns = {"base_model": load_base_model, "large_model": load_large_model}


def load_model(
    model_name: str, output_attentions: bool = False
) -> Tuple[T5EncoderModel, AutoTokenizer]:
    return available_models_fns[model_name](output_attentions=output_attentions)
