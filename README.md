# deep-significance: Easy Significance Testing for Deep Neural Networks

[![Build Status](https://travis-ci.com/Kaleidophon/deep-significance.svg?branch=main)]()
[![Coverage Status](https://coveralls.io/repos/github/Kaleidophon/deep-significance/badge.svg?branch=main&service=github)](https://coveralls.io/github/Kaleidophon/deep-significance?branch=main)
[![Compatibility](https://img.shields.io/badge/python-3.5%2B-blue)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

**Warning: This project is still under development. Code might be erroneous and breaking changes be introduced without 
warning.**

@TODO: Add shortcuts

### :interrobang: Why?

Although Deep Learning has undergone spectacular growth in the recent decade,
a large portion of experimental evidence is not supported by statistical hypothesis tests. Instead, 
conclusions are often drawn based on single performance scores. 

This is problematic: Neural network display highly non-convex
loss surfaces (Li et al., 2018) and their performance depends on the specific hyperparameters that were found, or stochastic factors 
like Dropout masks, making comparisons between architectures more difficult. Based on comparing only (the mean of) a 
few scores, **we often cannot 
conclude that one model type or algorithm is better than another**.
This endangers the progress in the field, as seeming success due to random chance might practitioners astray. 

For instance,
a recent study in Natural Language Processing by Narang et al. (2021) has found that many modifications proposed to 
transformers do not actually improve performance. Similar issues are known to plague other fields like e.g. 
Reinforcement Learning (Henderson et al., 2018) and Computer Vision (Borji, 2017) as well. 

To help mitigate this problem, this package supplies fully-tested re-implementations of useful functions for significance
testing:
* Non-parametric tests such as Almost Stochastic Order (Dror et al., 2019), bootstrap (Efron & Tibshirani, 1994) and 
  permutation-randomization.
* p-value corrections methods such as Bonferroni (Bonferroni, 1936) and Fisher (Fisher, 1992). 

All functions are fully tested and also compatible with common deep learning data structures, such as PyTorch / 
Tensorflow tensors as well as NumPy and Jax arrays.  For examples about the usage, consult the documentation 
[here](deep-significance.rtfd.io/en/latest/) or the scenarios in the section [Examples](#examples).

## :inbox_tray: Installation

(**The package has not been released on PyPI yet**)

The package can simply be installed using `pip` by running

    pip3 install deepsig

Another option is to clone the repository and install the package locally:

    git clone https://github.com/Kaleidophon/deep-significance.git
    cd deep-significance
    pip3 install -e .

**Warning**: Installed like this, imports will fail when the clones repository is moved.

## :bookmark: Examples

---
tl;dr: Use `aso()` to compare scores for two models. If the returned `eps_min < 0.5`, A is better than B. For
more info, check @TODO.

---

In the following, I will lay out three scenarios that describe common use cases for ML practitioners and how to apply 
the methods implemented in this package accordingly. For an introduction into statistical hypothesis testing, please
refer to resources such as [this blog post](https://machinelearningmastery.com/statistical-hypothesis-tests/) for a general
overview or [Dror et al. (2018)](https://www.aclweb.org/anthology/P18-1128.pdf) for a NLP-specific point of view. 

In general, in statistical significance testing, we usually compare to algorithms <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode" align=middle width=13.29340979999999pt height=22.465723500000017pt/> on a dataset <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> using 
some evaluation metric <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/b5eaea000e06d5cf2e882f8fdbc71e36.svg?invert_in_darkmode" align=middle width=19.740822749999992pt height=22.465723500000017pt/>. The difference between two algorithms on the data is then defined as 

<p align="center"><img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/9540dc879d2ecaa7cb245871b24f4e5d.svg?invert_in_darkmode" align=middle width=212.73480854999997pt height=16.438356pt/></p>

where <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/6dea53e880ae565b82d6b4a6148a0012.svg?invert_in_darkmode" align=middle width=35.622171749999985pt height=24.65753399999998pt/> is our test statistic. We then test the following **null hypothesis**:

<p align="center"><img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/1d210dbbb93bbdc5a632b9443059499d.svg?invert_in_darkmode" align=middle width=100.49629589999999pt height=16.438356pt/></p>

Thus, we assume our algorithm A to be equally as good or worse than algorithm B and reject the null hypothesis if A 
is better than B (what we is what we actually would like to see). Most statistical significance tests operate using 
**p-values**, which define the probability that under the null-hypothesis, the true difference <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/6dea53e880ae565b82d6b4a6148a0012.svg?invert_in_darkmode" align=middle width=35.622171749999985pt height=24.65753399999998pt/> is larger or e
equal than the observed difference <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/ecdae90a73f512871267f358443bd563.svg?invert_in_darkmode" align=middle width=26.32659479999999pt height=22.831056599999986pt/> (that is, for a one-sided test):

<p align="center"><img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/6d2735c4e335ec03c8b45736da4531a3.svg?invert_in_darkmode" align=middle width=135.91559685pt height=16.438356pt/></p>

Intuitively, the p-value is expressing: **How likely is it that the observed difference is up to what we expected, given that A is 
not better than B?** If this probability is high, it means that we're likely to see A is not better than B. If the 
probability is low, that means that <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/94ea44af3034479a1ba3f2f655bcec39.svg?invert_in_darkmode" align=middle width=26.32659479999999pt height=22.831056599999986pt/> is likely *larger* than <img src="https://rawgit.com/Kaleidophon/deep-significance/main/svgs/6dea53e880ae565b82d6b4a6148a0012.svg?invert_in_darkmode" align=middle width=35.622171749999985pt height=24.65753399999998pt/> - indicating 
that the null hypothesis might be wrong and that A is indeed better than B. 

To decide when we think A to be better than B, we typically set a confidence threshold, often 0.05.


### Intermezzo: Almost Stochastic Order - a better significance test for Deep Neural Networks

@TODO 


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

### :mortar_board: Cite

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

### :medal_sports: Acknowledgements

This package was created out of discussions of the [NLPnorth group](https://nlpnorth.github.io/) at the IT University 
Copenhagen. The code in this repository is in multiple places based on several of [Rotem Dror's](https://rtmdrr.github.io/) 
repositories, namely [this](https://github.com/rtmdrr/replicability-analysis-NLP), [this](https://github.com/rtmdrr/testSignificanceNLP)
and [this one](https://github.com/rtmdrr/DeepComparison).

The commit message template used in this project can be found [here](https://github.com/Kaleidophon/commit-template-for-humans).

### :books: Bibliography

Bonferroni, Carlo. "Teoria statistica delle classi e calcolo delle probabilita." Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commericiali di Firenze 8 (1936): 3-62.

Borji, Ali. "Negative results in computer vision: A perspective." Image and Vision Computing 69 (2018): 1-8.

Dror, Rotem, et al. "The hitchhiker’s guide to testing statistical significance in natural language processing." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018.

Dror, Rotem, Segev Shlomov, and Roi Reichart. "Deep dominance-how to properly compare deep neural models." Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019.

Efron, Bradley, and Robert J. Tibshirani. An introduction to the bootstrap. CRC press, 1994.

Fisher, Ronald Aylmer. "Statistical methods for research workers." Breakthroughs in statistics. Springer, New York, NY, 1992. 66-70.

Henderson, Peter, et al. "Deep reinforcement learning that matters." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein: Visualizing the Loss Landscape of Neural Nets. NeurIPS 2018: 6391-6401

Narang, Sharan, et al. "Do Transformer Modifications Transfer Across Implementations and Applications?." arXiv preprint arXiv:2102.11972 (2021).