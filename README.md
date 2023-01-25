<br/>

<h1 align="center">Ankh ☥: Optimized Protein Language Model Unlocks General-Purpose Modelling </h1>
<br/>

<br/>

[Ankh](https://arxiv.org/abs/2301.06568) is the first general-purpose protein language model trained on Google's **TPU-V4** surpassing the state-of-the-art performance with dramatically less parameters, promoting accessibility to research innovation via attainable resources.


<div align="center"><img width=500 height=350 src="https://github.com/agemagician/Ankh/blob/main/images/AnkhGIF.gif?raw=true"></div>




This repository will be updated regulary with **new pre-trained models for proteins** in part of supporting the **biotech** community in revolutinizing protein engineering using AI.


Table of Contents
=================
* [&nbsp; Installation](#install)
* [&nbsp; Models Availability](#models)
* [&nbsp; Dataset Availability](#datasets)
* [&nbsp; Usage](#usage)
* [&nbsp; Original downstream Predictions](#results)
* [&nbsp; Followup use-cases](#inaction)
* [&nbsp; Comparisons to other tools](#comparison)
* [&nbsp; Community and Contributions](#community)
* [&nbsp; Have a question?](#question)
* [&nbsp; Found a bug?](#bug)
* [&nbsp; Requirements](#requirements)
* [&nbsp; Sponsors](#sponsors)
* [&nbsp; Team](#team)
* [&nbsp; License](#license)
* [&nbsp; Citation](#citation)


<a name="install"></a>
## &nbsp; Installation

```python
python -m pip install ankh
```


<a name="models"></a>
## &nbsp; Models Availability

|               Model                |              ankh                 |                        Hugging Face                        |
|------------------------------------|-----------------------------------|-----------------------------------------------------------|
|             Ankh Large             |     `ankh.load_large_model()`     |[Ankh Large](https://huggingface.co/ElnaggarLab/ankh-large)| 
|             Ankh Base              |     `ankh.load_base_model()`      |[Ankh Base](https://huggingface.co/ElnaggarLab/ankh-base)  |


<a name="datasets"></a>

## &nbsp; Datasets Availability
|            Dataset            |                                            Hugging Face                                            |  
| ----------------------------- |---------------------------------------------------------------------------------------------------|
|	Remote Homology       	      |    `load_dataset("proteinea/remote_homology")`                                                    |
|	CASP12			                  |    `load_dataset("proteinea/secondary_structure_prediction", data_files={'test': ['CASP12.csv']})`|
|	CASP14			                  |    `load_dataset("proteinea/secondary_structure_prediction", data_files={'test': ['CASP14.csv']})`|
|	CB513			                    |    `load_dataset("proteinea/secondary_structure_prediction", data_files={'test': ['CB513.csv']})` |
|	TS115			                    |    `load_dataset("proteinea/secondary_structure_prediction", data_files={'test': ['TS115.csv']})` |
|	DeepLoc		                    |    `load_dataset("proteinea/deeploc")`                                                            |
| Fluorescence                  |    `load_dataset("proteinea/fluorescence")`                                                        |
| Solubility                    |    `load_dataset("proteinea/solubility")`                                                         |
| Nearest Neighbor Search       |    `load_dataset("proteinea/nearest_neighbor_search")`                                            |



<a name="usage"></a>
## &nbsp; Usage

* Loading pre-trained models:
```python
  import ankh

  # To load large model:
  model, tokenizer = ankh.load_large_model()
  model.eval()


  # To load base model.
  model, tokenizer = ankh.load_base_model()
  model.eval()
```

* Feature extraction using ankh large example:
```python

  model, tokenizer = ankh.load_large_model()
  model.eval()

  protein_sequences = ['MKALCLLLLPVLGLLVSSKTLCSMEEAINERIQEVAGSLIFRAISSIGLECQSVTSRGDLATCPRGFAVTGCTCGSACGSWDVRAETTCHCQCAGMDWTGARCCRVQPLEHHHHHH', 
  'GSHMSLFDFFKNKGSAATATDRLKLILAKERTLNLPYMEEMRKEIIAVIQKYTKSSDIHFKTLDSNQSVETIEVEIILPR']

  protein_sequences = [list(seq) for seq in protein_sequences]


  outputs = tokenizer.batch_encode_plus(protein_sequences, 
                                    add_special_tokens=True, 
                                    padding=True, 
                                    is_split_into_words=True, 
                                    return_tensors="pt")
  with torch.no_grad():
    embeddings = model(input_ids=outputs['input_ids'], attention_mask=outputs['attention_mask'])
```

* Loading downstream models example:
```python
  # To use downstream model for binary classification:
  binary_classification_model = ankh.ConvBertForBinaryClassification(input_dim=768, 
                                                                     nhead=4, 
                                                                     hidden_dim=384, 
                                                                     num_hidden_layers=1, 
                                                                     num_layers=1, 
                                                                     kernel_size=7, 
                                                                     dropout=0.2, 
                                                                     pooling='max')

  # To use downstream model for multiclass classification:
  multiclass_classification_model = ankh.ConvBertForMultiClassClassification(num_tokens=2, 
                                                                             input_dim=768, 
                                                                             nhead=4, 
                                                                             hidden_dim=384, 
                                                                             num_hidden_layers=1, 
                                                                             num_layers=1, 
                                                                             kernel_size=7, 
                                                                             dropout=0.2)

  # To use downstream model for regression:
  # training_labels_mean is optional parameter and it's used to fill the output layer's bias with it, 
  # it's useful for faster convergence.
  regression_model = ankh.ConvBertForRegression(input_dim=768, 
                                                nhead=4, 
                                                hidden_dim=384, 
                                                num_hidden_layers=1, 
                                                num_layers=1, 
                                                kernel_size=7, 
                                                dropout=0, 
                                                pooling='max', 
                                                training_labels_mean=0.38145)
```


<a name="results"></a>
## &nbsp; Original downstream Predictions 

<a name="q3"></a>
 * <b>&nbsp; Secondary Structure Prediction (Q3):</b><br/>
 
|         Model            |      CASP12      | CASP14 (HARD) |     TS115     |    CB513     |
|--------------------------|:----------------:|:-------------:|:-------------:|:------------:|
|**Ankh Large**            |      83.59%      |     77.48%    |    88.22%     |    88.48%    |
|Ankh Base                 |      80.81%      |     76.67%    |    86.92%     |    86.94%    |
|ProtT5-XL-UniRef50        |      83.34%      |     75.09%    |    86.82%     |    86.64%    |
|ESM2-15B                  |      83.16%      |     76.56%    |    87.50%     |    87.35%    |
|ESM2-3B                   |      83.14%      |     76.75%    |    87.50%     |    87.44%    |
|ESM2-650M                 |      82.43%      |     76.97%    |    87.22%     |    87.18%    |
|ESM-1b                    |      79.45%      |     75.39%    |    85.02%     |    84.31%    |


<a name="q8"></a>
 * <b>&nbsp; Secondary Structure Prediction (Q8):</b><br/>
 
|         Model            |      CASP12      | CASP14 (HARD) |     TS115     |    CB513     |
|--------------------------|:----------------:|:-------------:|:-------------:|:------------:|
|**Ankh Large**            |      71.69%      |     63.17%    |    79.10%     |    78.45%    |
|Ankh Base                 |      68.85%      |     62.33%    |    77.08%     |    75.83%    |
|ProtT5-XL-UniRef50        |      70.47%      |     59.71%    |    76.91%     |    74.81%    |
|ESM2-15B                  |      71.17%      |     61.81%    |    77.67%     |    75.88%    |
|ESM2-3B                   |      71.69%      |     61.52%    |    77.62%     |    75.95%    |
|ESM2-650M                 |      70.50%      |     62.10%    |    77.68%     |    75.89%    |
|ESM-1b                    |      66.02%      |     60.34%    |    73.82%     |    71.55%    |

<a name="CP"></a>
 * <b>&nbsp; Contact Prediction Long Precision Using Embeddings:</b><br/>
 
|         Model            | ProteinNet (L/1) | ProteinNet (L/5) | CASP14 (L/1)  | CASP14 (L/5) |
|--------------------------|:----------------:|:----------------:|:-------------:|:------------:|
|**Ankh Large**            |      48.93%      |      73.49%      |    16.01%     |    29.91%    |
|Ankh Base                 |      43.21%      |      66.63%      |    13.50%     |    28.65%    |
|ProtT5-XL-UniRef50        |      44.74%      |      68.95%      |    11.95%     |    24.45%    |
|ESM2-15B                  |      31.62%      |      52.97%      |    14.44%     |    26.61%    |
|ESM2-3B                   |      30.24%      |      51.34%      |    12.20%     |    21.91%    |
|ESM2-650M                 |      29.36%      |      50.74%      |    13.71%     |    22.25%    |
|ESM-1b                    |      29.25%      |      50.69%      |    10.18%     |    18.08%    |


<a name="CP"></a>
 * <b>&nbsp; Contact Prediction Long Precision Using attention scores:</b><br/>
 
|         Model            | ProteinNet (L/1) | ProteinNet (L/5) | CASP14 (L/1)  | CASP14 (L/5) |
|--------------------------|:----------------:|:----------------:|:-------------:|:------------:|
|**Ankh Large**            |      31.44%      |      55.58%      |     11.05%    |    20.74%    |
|Ankh Base                 |      25.93%      |      46.28%      |     9.32%     |    19.51%    |
|ProtT5-XL-UniRef50        |      30.85%      |      51.90%      |     8.60%     |    16.09%    |
|ESM2-15B                  |      33.32%      |      57.44%      |     12.25%    |    24.60%    |
|ESM2-3B                   |      33.92%      |      56.63%      |     12.17%    |    21.36%    |
|ESM2-650M                 |      31.87%      |      54.63%      |     10.66%    |    21.01%    |
|ESM-1b                    |      25.30%      |      42.03%      |     7.77%     |    15.77%    |


<a name="Loc"></a>
 * <b>&nbsp; Localization (Q10):</b><br/>
 
|         Model            |  DeepLoc Dataset |
|--------------------------|:----------------:|
|**Ankh Large**            |      83.01%      |
|Ankh Base                 |      81.38%      |
|ProtT5-XL-UniRef50        |      82.95%      |
|ESM2-15B                  |      81.22%      |
|ESM2-3B                   |      81.22%      |
|ESM2-650M                 |      82.08%      |
|ESM-1b                    |      80.51%      |


<a name="RH"></a>
 * <b>&nbsp; Remote Homology:</b><br/>
 
|         Model            |   SCOPe (Fold)   |
|--------------------------|:----------------:|
|Ankh Large                |      61.01%      |
|**Ankh Base**             |      61.14%      |
|ProtT5-XL-UniRef50        |      59.38%      |
|ESM2-15B                  |      54.48%      |
|ESM2-3B                   |      59.24%      |
|ESM2-650M                 |      51.36%      |
|ESM-1b                    |      56.93%      |


<a name="Sol"></a>
 * <b>&nbsp; Solubility:</b><br/>
 
|         Model            |    Solubility    |
|--------------------------|:----------------:|
|**Ankh Large**            |      76.41%      |
|Ankh Base                 |      76.36%      |
|ProtT5-XL-UniRef50        |      76.26%      |
|ESM2-15B                  |      60.52%      |
|ESM2-3B                   |      74.91%      |
|ESM2-650M                 |      74.56%      |
|ESM-1b                    |      74.91%      |


<a name="Flu"></a>
 * <b>&nbsp; Fluorescence (Spearman Correlation):</b><br/>
 
|         Model            |   Fluorescence   |
|--------------------------|:----------------:|
|**Ankh Large**            |        0.62      |
|Ankh Base                 |        0.62      |
|ProtT5-XL-UniRef50        |        0.61      |
|ESM2-15B                  |        0.56      |
|ESM-1b                    |        0.48      |
|ESM2-650M                 |        0.48      |
|ESM2-3B                   |        0.46      |


<a name="CATH"></a>
 * <b>&nbsp; Nearest Neighbor Search using Global Pooling:</b><br/>
 
|         Model            |   Lookup69K (C)  |   Lookup69K (A)  |   Lookup69K (T)  |   Lookup69K (H)  |
|--------------------------|:----------------:|:----------------:|:----------------:|:----------------:|
|Ankh Large                |       0.83       |       0.72       |       0.60       |       0.70       |
|**Ankh Base**             |       0.85       |       0.77       |       0.63       |       0.72       |
|ProtT5-XL-UniRef50        |       0.83       |       0.69       |       0.57       |       0.73       |
|ESM2-15B                  |       0.78       |       0.63       |       0.52       |       0.67       |
|ESM2-3B                   |       0.79       |       0.65       |       0.53       |       0.64       |
|ESM2-650M                 |       0.72       |       0.56       |       0.40       |       0.53       |
|ESM-1b                    |       0.78       |       0.65       |       0.51       |       0.63       |



<a name="team"></a>
## &nbsp; Team

* <b>Technical University of Munich:</b><br/>

| [Ahmed Elnaggar](https://github.com/agemagician) |       Burkhard Rost       |
|:------------------------------------------------:|:-------------------------:|
| <img width=120 src="https://github.com/agemagician/Ankh/blob/main/images/AhmedElNaggar.jpg?raw=true"> | <img width=120 src="https://github.com/agemagician/Ankh/blob/main/images/Rost.jpg?raw=true"> |


* <b>Proteinea:</b><br/>

| [Hazem Essam](https://github.com/hazemessamm) | [Wafaa Ashraf](https://github.com/wafaaashraf) | [Walid Moustafa](https://github.com/wmustafaawad) | [Mohamed Elkerdawy](https://github.com/melkerdawy) |
|:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:|
| <img width=120 src="https://github.com/agemagician/Ankh/blob/main/images/HazemEssam.jpeg?raw=true"> | <img width=120 src="https://github.com/agemagician/Ankh/blob/main/images/WafaaAshraf.jpeg?raw=true"> | <img width=120 src="https://github.com/agemagician/Ankh/blob/main/images/WalidMoustafa.jpg?raw=true"> | <img width=120 src="https://github.com/agemagician/Ankh/blob/main/images/MohamedElKerdawy.jpeg?raw=true"> |


* <b>University of Columbia:</b><br/>

| [Charlotte Rochereau](https://github.com/crochereau) |
|:----------------------------------------------------:|
| <img width=120 src="https://github.com/agemagician/Ankh/blob/main/images/CharlotteRochereau.jpg?raw=true"> |


<a name="sponsors"></a>
## &nbsp; Sponsors


|                                                    Google Cloud                                                         |
:------------------------------------------------------------------------------------------------------------------------:|
<img width=120 src="https://github.com/agemagician/Ankh/blob/main/images/google_cloud_logo.jpg?raw=true"> |



<a name="license"></a>
## &nbsp; License
Ankh pretrained models are released under the under terms of the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International](https://creativecommons.org/licenses/by/4.0/).

<a name="community"></a>
## &nbsp; Community and Contributions

The Ankh project is a **open source project** supported by various partner companies and research institutions. We are committed to **share all our pre-trained models and knowledge**. We are more than happy if you could help us on sharing new ptrained models, fixing bugs, proposing new feature, improving our documentation, spreading the word, or support our project.

<a name="question"></a>
## &nbsp; Have a question?

We are happy to hear your question in our issues page [Ankh](https://github.com/agemagician/Ankh/issues)! Obviously if you have a private question or want to cooperate with us, you can always **reach out to us directly** via [Hello](mailto:hello@proteinea.com?subject=[GitHub]Ankh). 

<a name="bug"></a>
## &nbsp; Found a bug?

Feel free to **file a new issue** with a respective title and description on the the [Ankh](https://github.com/agemagician/Ankh/issues) repository. If you already found a solution to your problem, **we would love to review your pull request**!.

<a name="citation"></a>
## ✏️&nbsp; Citation
If you use this code or our pretrained models for your publication, please cite the original paper:
```
@article{elnaggar2023ankh,
  title={Ankh: Optimized Protein Language Model Unlocks General-Purpose Modelling},
  author={Elnaggar, Ahmed and Essam, Hazem and Salah-Eldin, Wafaa and Moustafa, Walid and Elkerdawy, Mohamed and Rochereau, Charlotte and Rost, Burkhard},
  journal={arXiv preprint arXiv:2301.06568},
  year={2023}
}
```
