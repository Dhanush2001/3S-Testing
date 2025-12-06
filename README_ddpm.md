# Can you rely on your model evaluation? Improving model evaluation with synthetic test data

[![arXiv](https://img.shields.io/badge/arXiv--b31b1b.svg)](https://arxiv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/vanderschaarlab/3S-Testing/blob/main/LICENSE)

This repository contains code for the paper "Can you rely on your model evaluation? Improving model evaluation with synthetic test data".

For more details, please read our [NeurIPS 2023 paper](https://arxiv.org).

---

## ⚠ Important Note  
To use the **DDPM implementation**, switch to the dedicated branch:

```shell
git checkout ddpm
```

All DDPM-related files and training scripts are available only in this branch.

---

## Installation

1. Clone the repository

2. Create a new conda environment with Python 3.9, for example:

```shell
conda create --name 3s_env python=3.9
```

3. Install DDPM-specific requirements:

```shell
pip install -r requirements_ddpm.txt
```

4. Link the environment to Jupyter:

```shell
python -m ipykernel install --user --name=3s_env
```

---

## Use-cases

We highlight different use-cases of 3S-Testing for subgroup analysis and shift testing in the notebooks found in the `/use_cases` folder.

---

## Citing

If you use this code, please cite the associated paper:

```
@inproceedings{3STesting,
title={Can you rely on your model evaluation? Improving model evaluation with synthetic test data},
author={van Breugel, Boris and Seedat, Nabeel and Imrie, Fergus and van der Schaar, Mihaela},
booktitle={Advances in Neural Information Processing Systems},
year={2023}
}
```
