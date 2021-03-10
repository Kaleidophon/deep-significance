# deep-significance: Easy Significance Testing for Deep Neural Networks

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

**Warning: This project is still under development. Code might be erroneous and breaking changes be introduced without 
warning.**

### |:interrobang:| Why?

Although the field of Machine Learning and Deep Learning has undergone spectacular growth in the recent decade,
a large portion of experimental evidence is not supported by statistical significance tests. Instead, 
conclusions are often drawn based on single performance scores. 

This is problematic: Neural network display highly non-convex
loss surfaces (Li et al., 2018) and their performance depends on the specific hyperparameters that were found, or stochastic factors 
like Dropout masks, making comparisons more difficult. Based on comparing only (the mean of) a few scores, **we cannot conclude that one model type is better than another**.
This endangers the progress in the field, as seeming success due to random chance might lead practicioners astray. 
For instance,
a recent study in Natural Language Processing by Narang et al. (2021) has found that many modifications proposed to 
transformers do not actually improve performance. Similar issues are known to plague other fields like e.g. 
Reinforcement Learning (Henderson et al., 2018) and Computer Vision (Borji, 2017) as well. 

To help mitigate this problem, this package supplies fully-tested reimplementations of useful functions for significance
testing:
* Non-parametic test such as Almost Stochastic Order (Dror et al., 2019), bootstrap (Efron & Tibshirani, 1994) and 
  permutation-randomization.
* p-value corrections methods such as Bonferroni (Bonferroni, 1936) and Fisher (Fisher, 1992). 

All functions are fully tested and also compatible with common deep learning data structures, such as PyTorch and 
Tensorflow tensors as well as NumPy and Jax arrays.  For examples about the usage, consult the documentation here 
(@TODO: Add link to docs) or the scenarios in the section [Examples](#examples).

## |:inbox_tray:| Installation

(**The package has not been released on pip yet**)

The package can simply be installed using `pip` by running

    pip3 install deepsig

### |:bulb:| A short and gentle introduction to significance testing (for DNNs)

@TODO: Significant testing basic idea
@TODO: Multiple comparisons
@TODO: Significance testing for neural networks

## |:bookmark:| Examples

In the following, I will lay out three scenarios that describe common use cases for ML practitioners and how to apply 
the methods implemented in this package accordingly.

### Scenario 1 - Comparing multiple runs of two models 

In the simplest scenario, we have retrieved a set of scores from a model A and a baseline B on a dataset, stemming from 
various model runs with different seeds. We can now simply apply the ASO test:

```python
from deepsig import aso

scores_a = ...  # TODO
scores_b = ...  # TODO

min_eps = aso(scores_a, scores_b)  # min_eps = ..., so A is better
```

Because ASO is a non-parametric test, **it does not make any assumptions about the distributions of the scores**. 
This means that we can apply it to any kind of test metric. The scores of model runs are supplied, the more reliable 
the test becomes.

### Scenario 2 - Comparing multiple runs across datasets

@TODO: Comparison between two models, multiple datasets

### Scenario 3 - Comparing sample-level scores

@TODO: Comparison between two models, multiple seeds, sample-level

### |:mortar_board:| Cite

If you use the ASO test via `aso()`, please cite the original work:

    @inproceedings{dror2019deep,
      author    = {Rotem Dror and
                   Segev Shlomov and
                   Roi Reichart},
      editor    = {Anna Korhonen and
                   David R. Traum and
                   Llu{\'{\i}}s M{\`{a}}rquez},
      title     = {Deep Dominance - How to Properly Compare Deep Neural Models},
      booktitle = {Proceedings of the 57th Conference of the Association for Computational
                   Linguistics, {ACL} 2019, Florence, Italy, July 28- August 2, 2019,
                   Volume 1: Long Papers},
      pages     = {2773--2785},
      publisher = {Association for Computational Linguistics},
      year      = {2019},
      url       = {https://doi.org/10.18653/v1/p19-1266},
      doi       = {10.18653/v1/p19-1266},
      timestamp = {Tue, 28 Jan 2020 10:27:52 +0100},
    }

### |:medal_sports:| Credit

@TODO

### |:books:| Bibliography

Bonferroni, Carlo. "Teoria statistica delle classi e calcolo delle probabilita." Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commericiali di Firenze 8 (1936): 3-62.

Borji, Ali. "Negative results in computer vision: A perspective." Image and Vision Computing 69 (2018): 1-8.

Dror, Rotem, Segev Shlomov, and Roi Reichart. "Deep dominance-how to properly compare deep neural models." Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019.

Efron, Bradley, and Robert J. Tibshirani. An introduction to the bootstrap. CRC press, 1994.

Fisher, Ronald Aylmer. "Statistical methods for research workers." Breakthroughs in statistics. Springer, New York, NY, 1992. 66-70.

Henderson, Peter, et al. "Deep reinforcement learning that matters." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein: Visualizing the Loss Landscape of Neural Nets. NeurIPS 2018: 6391-6401

Narang, Sharan, et al. "Do Transformer Modifications Transfer Across Implementations and Applications?." arXiv preprint arXiv:2102.11972 (2021).