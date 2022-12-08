from .models import get_available_models, load_base_model, load_large_model, load_model

from .utils import FastaDataset, CSVDataset

from .models import (
    ConvBertForBinaryClassification,
)
from .models import (
    ConvBertForMultiClassClassification,
)
from .models import ConvBertForRegression
from typing import Union


available_tasks = {
    "binary": ConvBertForBinaryClassification,
    "regression": ConvBertForRegression,
    "multiclass": ConvBertForMultiClassClassification,
}


def get_available_tasks():
    return list(available_tasks.keys())


def load_downstream_model(
    task,
) -> Union[
    ConvBertForBinaryClassification,
    ConvBertForMultiClassClassification,
    ConvBertForRegression,
]:
    return available_tasks[task]


__version__ = "1.0"
