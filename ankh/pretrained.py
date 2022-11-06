from transformers import T5EncoderModel, AutoTokenizer
from enum import Enum
import os


supported_models = {
    'base_model': "ElnaggarLab/protx-base-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel768",
    'large_model': "ElnaggarLab/protx-large-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel1536"
}

class AvailableModels(Enum):
    BASE_MODEL = "ElnaggarLab/protx-base-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel768"
    LARGE_MODEL = "ElnaggarLab/protx-large-1gspan-partreconstruction-20mlmp-encl48-decl24-ramd128-ranb64-dmodel1536"

def get_available_models():
    return list(supported_models.keys())

def load_base_model(output_attentions=False):
    tokenizer = AutoTokenizer.from_pretrained(supported_models['base_model'], use_auth_token=os.environ['huggingface_token'])
    model = T5EncoderModel.from_pretrained(supported_models['base_model'], use_auth_token=os.environ['huggingface_token'], output_attentions=output_attentions)
    return model, tokenizer

def load_large_model(output_attentions=False):
    tokenizer = AutoTokenizer.from_pretrained(supported_models['base_model'], use_auth_token=os.environ['huggingface_token'])
    model = T5EncoderModel.from_pretrained(supported_models['large_model'], use_auth_token=os.environ['huggingface_token'], output_attentions=output_attentions)
    return model, tokenizer

supported_models_fns = {
    'base_model': load_base_model,
    'large_model': load_large_model
}

def load_model(model_name, output_attentions=False):
    return supported_models_fns[model_name](output_attentions=output_attentions)