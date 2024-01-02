# CONAN
Source code for paper:[Building A Coding Assistant via Retrieval-Augmented Language Models]()



## CONAN-R

CONAN-R使用[**SANTA**](https://github.com/OpenMatch/SANTA)初始化，然后在每个生成任务的训练集上进行微调。

### 1.Requirements

(1) Install the following packages using Pip or Conda under this environment:

```
transformers==4.22.2    # 其他版本可能会出现错误。
```

(2) install openmatch. To download OpenMatch as a library and obtain openmatch-thunlp-0.0.1.

```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install .
```



### 2.获取SANTA Checkpoint

The checkpoint of the pretrained SANTA model on `Python` data is [here](https://huggingface.co/OpenMatch/santa-code-python-adv).



### 3.准备微调数据

因为我们使用OpenMatch微调CONAN-R，所以首先我们需要将数据整理成OpenMatch能使用的格式。

1）build code





2）finetune





## CONAN-G

