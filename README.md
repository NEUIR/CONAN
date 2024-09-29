# CONAN
Source code for paper:[Building A Coding Assistant via Retrieval-Augmented Language Models](https://dl.acm.org/doi/10.1145/3695868)

### Overview
Pretrained language models have shown strong effectiveness in code-related tasks, such as code retrieval, code generation, code summarization, and code completion tasks. In this paper, we propose **CO**de assista**N**t vi**A** retrieval-augme**N**ted language model (CONAN), which aims to build a code assistant by mimicking the knowledge-seeking behaviors of humans during coding. Specifically, it consists of a code structure aware retriever (CONAN-R) and a dual-view code representation-based retrieval-augmented generation model (CONAN-G). CONAN-R pretrains CodeT5 using Code-Documentation Alignment and Masked Entity Prediction tasks to make language models code structure-aware and learn effective representations for code snippets and documentation. Then CONAN-G designs a dual-view code representation mechanism for implementing a retrieval-augmented code generation model. CONAN-G regards the code documentation descriptions as prompts, which help language models better understand the code semantics. Our experiments show that CONAN achieves convincing performance on different code generation tasks and significantly outperforms previous retrieval augmented code generation models. Our further analyses show that CONAN learns tailored representations for both code snippets and documentation by aligning code-documentation data pairs and capturing structural semantics by masking and predicting entities in the code data. Additionally, the retrieved code snippets and documentation provide necessary information from both program language and natural language to assist the code generation process. CONAN can also be used as an assistant for Large Language Models (LLMs), providing LLMs with external knowledge in shorter code document lengths to improve their effectiveness on various code tasks. It shows the ability of CONAN to extract necessary information and help filter out the noise from retrieved code documents.

### Reproduce CONAN
CONAN has two parts. You can use them seperately as well.

- CONAN-R: Please see instructions in [./CONAN-R](CONAN-R/README.md).
- CONAN-G: Please see instructions in [./CONAN-G](CONAN-G/README.md).

### SANTA
CONAN is an extension of [SANTA](https://arxiv.org/abs/2305.19912) on ACL 2023. The previous conference version focused only on learning the representation of structured data to improve the performance of code retrieval. However, most of the existing code retrieval systems are combined with code generation tasks such as code generation, code summarization, and code completion to build code retrieval augmented frameworks. The knowledge boundary problem of the language model can be alleviated by retrieving relevant code snippets and documentation from external knowledge bases. Therefore, based on the previous work, we have made the following improvements and extensions: 1) we extend our previous SANTA model into a code assistant (CONAN), which consists of a code structure-aware retriever and a dual-view code representation-based retrieval-augmented generation
model. 2) The dual-view code representation-based retrieval-augmented generation model designs a dual-view code representation mechanism that helps language models better understand code semantics by regarding the code documentation descriptions as prompts. 3) The dual-view code representation-based retrieval-augmented generation model employs the Fusion in Decoder (FID) architecture, which breaks the limitation of the input length of the language model. 4) The code assistant (CONAN) performs well on several code-related tasks including code retrieval, code generation, code summarization and code completion. 5) CONAN can be used as an assistant for the large language models to assist them in finishing various code tasks. 

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






