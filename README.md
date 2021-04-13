# On the Inductive Bias of Masked Language Modeling

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

#### Authors
* [Tianyi Zhang](https://tiiiger.github.io/)
* [Tatsunori Hashimoto](https://thashim.github.io/)

## Overview
This repo containst the official code release for NAACL 2021 paper _On the Inductive Bias of Masked Language Modeling: From Statistical to Syntactic Dependencies_.

If you find this repo useful, please cite: 
```
@InProceedings{naacl21zhang,
  title = 	 {On the Inductive Bias of Masked Language Modeling: From Statistical to Syntactic Dependencies},
  author = 	 {Zhang, Tianyi and Hashimoto, Tatsunori},
  booktitle	=   {North American Association for Computational Linguistics (NAACL)},
  year = 	 {2021},
}
```

We organize the code to replicate different experiments into subfolders.
The experiments in section 3.2 and 5.1 are in `./finetuning` and the experiment in section 5.2 are in `./unsupervised_parsing`.

## Dependencies
`requirements.txt` lists the essential requirements for this project.
