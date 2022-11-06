from ankh.downstream_models.convbert_binary_classification import ConvBertForBinaryClassification
from ankh.downstream_models.convbert_multiclass_classification import ConvBertForMultiClassClassification
from ankh.downstream_models.convbert_regression import ConvBertForRegression


supported_tasks = {
    'binary': ConvBertForBinaryClassification,
    'regression': ConvBertForRegression,
    'multiclass': ConvBertForMultiClassClassification,
}

def get_supported_tasks():
    return list(supported_tasks.keys())

def load_downstream_model(task, **kwargs):
    return supported_tasks[task](**kwargs)
