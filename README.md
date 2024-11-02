# CONAN
[TOIS 2024] This is source code for paper:[Building A Coding Assistant via Retrieval-Augmented Language Models](https://dl.acm.org/doi/10.1145/3695868)

## Overview
We propose **CO**de assista**N**t vi**A** retrieval-augme**N**ted language model (CONAN), which aims to build a code assistant by mimicking the knowledge-seeking behaviors of humans during coding. Specifically, it consists of a code structure aware retriever (CONAN-R) and a dual-view code representation-based retrieval-augmented generation model (CONAN-G). CONAN-R pretrains CodeT5 using Code-Documentation Alignment and Masked Entity Prediction tasks to make language models code structure-aware and learn effective representations for code snippets and documentation. Then CONAN-G designs a dual-view code representation mechanism for implementing a retrieval-augmented code generation model. CONAN-G regards the code documentation descriptions as prompts, which help language models better understand the code semantics.

## Reproduce CONAN
CONAN consists of two parts, CONAN-R and CONAN-G. You can use them seperately as well.

### CONAN-R
CONAN-R use CodeT5-base initialization, and then fine-tune on each task of training set.  

#### 1.Requirements

(1) Creating a virtual Environment

```bash
conda create -n conanr python=3.8
```

(2) Install the following packages using Pip or Conda under this environment:

```
transformers==4.22.2    # Other versions may have errors.
datasets
Pillow
torch

tensorboard
```
(3) install openmatch. To download OpenMatch as a library and obtain openmatch-thunlp-0.0.1.

```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install .
```

#### 2.Get SANTA Checkpoint

The checkpoint of the pretrained CONAN-R model on `Python` data is [here](https://huggingface.co/OpenMatch/santa-code-python-adv). The pre-training code and README file for CONANA-R can be found [here](https://github.com/OpenMatch/SANTA).


#### 3.Finetune 

After pre-training CONAN-R, you can use the following process to fine-tune CONAN-R on the appropriate retrieval corpus.

(1) download dataset**
You can download the data for the experiment from [REDCODER](https://arxiv.org/abs/2305.19912) and [REACC](https://github.com/microsoft/ReACC)


(2) build code

Convert the dataset to the format required by OpenMatch. Open ```CONAN-R/build_code``` and run ```build_code.sh```. This process allows you to build queries in REDCODER and REACC into the format used by openmatch.

```bash
bash build_code.sh
```

(3) finetune
You can use the ```CONAN-R/OpenMatch/train.sh``` script to fine-tune the retriever on each task's dataset to get CONAN-R. 
```bash
bash train.sh
```

(4) inference
After obtaining the fine-tuned CONAN-R, you need to use the fine-tuned CONAN-R to encode the code retrieval corpus. 
For different tasks and datasets, you need to use the CONAN-R fine-tuned on the corresponding datasets to encode the relevant corpora. Open ```CONAN-R/infer``` and run the script on the corresponding dataset. 
(5) search
After obtaining the embeddings for the corpora corresponding to each task, you need to retriever the code documents for the respective tasks. You can open folder ```search``` and run the script to retrieve the corresponding augmentation code documents for different training and testing datasets.

### CONAN-G
CONAN-G use CodeT5-base initialization, and finetune using the training dataset inferred from CONAN-R in the previous step.

#### 1.Requirements

(1) Creating a virtual Environment

```bash
conda create -n conang python=3.8
```

(2) Install packages 

```sh
cd CONAN-G
pin install -r requirements.txt
```
#### 2.Training 
You need to use the relevant training data retrieved by CONAN-R to fine-tune the CONAN-G model. Open ```CONAN-G``` and run ```train_reader.sh``` for the corresponding dataset.

```bash
bash train_reader.sh
```

#### 3.Evaluation

After training CONAN-G, you need to use CONAN-R to retrieve the augmentation code documents for the test datasets. Then, Open ```CONAN-G``` and run ```test_reader.sh``` to test the performance of CONAN-G on these retrieved test datasets.
```bash
bash test_reader.sh
```

### Citation
If you find this paper or this code useful, please cite this paper:

```
@article{10.1145/3695868,
author = {Li, Xinze and Wang, Hanbin and Liu, Zhenghao and Yu, Shi and Wang, Shuo and Yan, Yukun and Fu, Yukai and Gu, Yu and Yu, Ge},
title = {Building A Coding Assistant via the Retrieval-Augmented Language Model},
year = {2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
issn = {1046-8188},
url = {https://doi.org/10.1145/3695868},
doi = {10.1145/3695868},
note = {Just Accepted},
journal = {ACM Trans. Inf. Syst.},
month = sep,
keywords = {Code Assistant, Code Generation, Code Retrieval, Retrieval Augmented Language Model}
}
```






