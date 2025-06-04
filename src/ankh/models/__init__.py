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
from ankh.models.contact_prediction import ConvBERTForContactPrediction
from ankh.models.contact_prediction import ContactPredictionHead
from ankh.models.ankh_transformers import (
    get_available_models,
    load_ankh_base,
    load_ankh_large,
    load_ankh3_large,
    load_ankh3_xl,
    load_model,
)

# Aliases for backward compatibility.
# Deprecated, will be removed in the future.
ConvBertForBinaryClassification = ConvBERTForBinaryClassification
ConvBertForMultiClassClassification = ConvBERTForMultiClassClassification
ConvBertForMultiLabelClassification = ConvBERTForMultiLabelClassification
ConvBertForRegression = ConvBERTForRegression
ConvBertForContactPrediction = ConvBERTForContactPrediction
load_large_model = load_ankh_large
load_base_model = load_ankh_base
