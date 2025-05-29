from ankh.models.convbert_binary_classification import (
    ConvBERTForBinaryClassification,
)
from ankh.models.convbert_multiclass_classification import (
    ConvBERTForMultiClassClassification,
)
from ankh.models.convbert_multilabel_classification import (
    ConvBERTForMultiLabelClassification,
)
from ankh.models.convbert_regression import ConvBERTForRegression
from ankh.models.ankh_transformers import (
    get_available_models,
    load_ankh_base,
    load_ankh_large,
    load_ankh3_large,
    load_ankh3_xl,
    load_model,
    load_large_model,  # deprecated
    load_base_model,  # deprecated
)

# Aliases for backward compatibility.
# Deprecated, will be removed in the future.
ConvBertForBinaryClassification = ConvBERTForBinaryClassification
ConvBertForMultiClassClassification = ConvBERTForMultiClassClassification
ConvBertForMultiLabelClassification = ConvBERTForMultiLabelClassification
ConvBertForRegression = ConvBERTForRegression
