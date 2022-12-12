<br/>
<h1 align="center">Ankh: Optimized Protein Language Model</h1>
<br/>

<br/>

[Ankh PLM](https://github.com/agemagician/Ankh/) is providing **state of the art pre-trained models for proteins**. Ankh was trained on **TPU V4-128**.

Have a look at our paper [Placeholder](paperlink) for more information about our work. 

<br/>
<p align="center">
    <img width="70%" src="https://github.com/agemagician/ProtTrans/raw/master/images/transformers_attention.png" alt="ProtTrans Attention Visualization">
</p>
<br/>


This repository will be updated regulary with **new pre-trained models for proteins** as part of supporting **bioinformatics** community in general.


Table of Contents
=================
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

<table>
  <tr>
    <th>Models</th>
    <th>Dataset</th>
    <th>HuggingFace</th>
    <th>Ankh</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Ankh Large</td>
    <td>Uniref 50</td>
    <td>placeholder</td>
    <td>```python ankh.load_large_model() ```</td>
    <td>placeholder</td>
  </tr>
  <tr>
    <td>Ankh Base</td>
    <td>Uniref 50</td>
    <td>placeholder</td>
    <td>ankh.load_base_model()</td>
    <td>placeholder</td>
  </tr>
</table>


<a name="datasets"></a>
## ⌛️&nbsp; Datasets Availability


<table>
  <tr>
    <th>Dataset Name</th>
    <th>HuggingFace</th>
    <th>Description</th>
  </tr>
  <tr>
    <td>Secondary Structure Prediction</td>
    <td>`load_dataset('proteinea/SSP')`</td>
    <td>placeholder</td>
  </tr>
  <tr>
    <td>Fluorosence</td>
    <td>`load_dataset('proteinea/Fluorosence')`</td>
    <td>placeholder</td>
  </tr>
  <tr>
    <td>Solubility</td>
    <td>`load_dataset('proteinea/Solubility')`</td>
    <td>placeholder</td>
  </tr>
  <tr>
    <td>Solubility</td>
    <td>`load_dataset('proteinea/Solubility')`</td>
    <td>placeholder</td>
  </tr>
</table>

