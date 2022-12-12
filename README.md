<br/>
<h1 align="center">Ankh: Optimized Protein Language Model</h1>
<br/>

<br/>

[Ankh PLM](https://github.com/agemagician/Ankh/) is providing **state of the art pre-trained models for proteins**. Ankh was trained on **TPU-V4**.

Have a look at our paper [Placeholder](paperlink) for more information about our work. 




This repository will be updated regulary with **new pre-trained models for proteins** as part of supporting **bioinformatics** community in general.


Table of Contents
=================
* [ ⌛️&nbsp; News](#news)
* [ ⌛️&nbsp; Models Availability](#models)
* [ ⌛️&nbsp; Dataset Availability](#datasets)
* [ 🚀&nbsp; Usage ](#usage)
  * [ 🧬&nbsp; Feature Extraction (FE)](#feature-extraction)
  * [ 🚀&nbsp; Logits extraction](#logits-extraction)
  * [ 💥&nbsp; Fine Tuning (FT)](#fine-tuning)
  * [ 🧠&nbsp; Prediction](#prediction)
  * [ ⚗️&nbsp; Protein Sequences Generation ](#protein-generation)
  * [ 🧐&nbsp; Visualization ](#visualization)
  * [ 📈&nbsp; Benchmark ](#benchmark)
* [ 📊&nbsp; Original downstream Predictions  ](#results)
* [ 📊&nbsp; Followup use-cases  ](#inaction)
* [ 📊&nbsp; Comparisons to other tools ](#comparison)
* [ ❤️&nbsp; Community and Contributions ](#community)
* [ 📫&nbsp; Have a question? ](#question)
* [ 🤝&nbsp; Found a bug? ](#bug)
* [ ✅&nbsp; Requirements ](#requirements)
* [ 🤵&nbsp; Team ](#team)
* [ 💰&nbsp; Sponsors ](#sponsors)
* [ 📘&nbsp; License ](#license)
* [ ✏️&nbsp; Citation ](#citation)


<a name="models"></a>
## ⌛️&nbsp; Models Availability

|               Model                |              ankh                 |              Hugging Face             |
|------------------------------------|-----------------------------------|---------------------------------------|
|             Ankh Large             |     `ankh.load_large_model()`     |          [Download](placeholder)      | 
|             Ankh Base              |     `ankh.load_base_model()`      |          [Download](placeholder)      |


<a name="datasets"></a>
## ⌛️&nbsp; Datasets Availability
|          Dataset              |                                    HuggingFace                             |  
| ----------------------------- |----------------------------------------------------------------------------|
|	Remote Homology       	    |    `load_dataset("proteinea/remote_homology")`                             |
|	CASP12			            |    `load_dataset("proteinea/SSP", data_files={'test': ['CASP12.csv']})`    |
|	CASP14			            |    `load_dataset("proteinea/SSP", data_files={'test': ['CASP14.csv']})`    |
|	CB513			            |    `load_dataset("proteinea/SSP", data_files={'test': ['CB513.csv']})`     |
|	TS115			            |    `load_dataset("proteinea/SSP", data_files={'test': ['TS115.csv']})`     |
|	DeepLoc		                |    `load_dataset("proteinea/deeploc")`                                     |
|   Fluorosence                 |    `load_dataset("proteinea/flourosence")`                                 |
|   Solubility                  |    `load_dataset("proteinea/solubility")`                                  |
|   Nearest Neighbor Search     |    `load_dataset("proteinea/nearest_neighbor_search")`                     |
