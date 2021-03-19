# deep-significance: Easy Significance Testing for Deep Neural Networks

[![Build Status](https://travis-ci.com/Kaleidophon/deep-significance.svg?branch=main)]()
[![Coverage Status](https://coveralls.io/repos/github/Kaleidophon/deep-significance/badge.svg?branch=main&service=github)](https://coveralls.io/github/Kaleidophon/deep-significance?branch=main)
[![Compatibility](https://img.shields.io/badge/python-3.5%2B-blue)]()
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

---
**Warning: This project is still under development. Code might be erroneous and breaking changes be introduced without 
warning.** Test

---

**Contents**

* [:interrobang: Why](#interrobang-why)
* [:inbox_tray: Installation](#inbox_tray-installation)
* [:bookmark: Examples](#bookmark-examples)
  * [Intermezzo: Almost Stochastic Order - a better significance test for Deep Neural Networks](#intermezzo-almost-stochastic-order---a-better-significance-test-for-deep-neural-networks)
  * [Scenario 1: Comparing multiple runs of two models](#scenario-1---comparing-multiple-runs-of-two-models)
  * [Scenario 2: Comparing multiple runs across datasets](#scenario-2---comparing-multiple-runs-across-datasets) 
  * [Scenario 3: Comparing sample-level scores](#scenario-3---comparing-sample-level-scores)
  * [Scenario 4: Comparing more than two models](#scenario-4---comparing-more-than-two-models)
  * [General Recommendations & other notes](#general-recommendations) 
* [:mortar_board: Cite](#mortar_board-cite)
* [:medal_sports: Acknowledgements](#medal_sports-acknowledgements)
* [:books: Bibliography](#books-bibliography)

### :interrobang: Why?

Although Deep Learning has undergone spectacular growth in the recent decade,
a large portion of experimental evidence is not supported by statistical hypothesis tests. Instead, 
conclusions are often drawn based on single performance scores. 

This is problematic: Neural network display highly non-convex
loss surfaces (Li et al., 2018) and their performance depends on the specific hyperparameters that were found, or stochastic factors 
like Dropout masks, making comparisons between architectures more difficult. Based on comparing only (the mean of) a 
few scores, **we often cannot 
conclude that one model type or algorithm is better than another**.
This endangers the progress in the field, as seeming success due to random chance might draw practitioners astray. 

For instance,
a recent study in Natural Language Processing by Narang et al. (2021) has found that many modifications proposed to 
transformers do not actually improve performance. Similar issues are known to plague other fields like e.g., 
Reinforcement Learning (Henderson et al., 2018) and Computer Vision (Borji, 2017) as well. 

To help mitigate this problem, this package supplies fully-tested re-implementations of useful functions for significance
testing:
* Statistical Significane tests such as Almost Stochastic Order (Dror et al., 2019), bootstrap (Efron & Tibshirani, 1994) and 
  permutation-randomization (Noreen, 1989).
* Bonferroni correction methods for multiplicity in datasets (Bonferroni, 1936). 

All functions are fully tested and also compatible with common deep learning data structures, such as PyTorch / 
Tensorflow tensors as well as NumPy and Jax arrays.  For examples about the usage, consult the documentation 
[here](deep-significance.rtfd.io/en/latest/) or the scenarios in the section [Examples](#examples).

## :inbox_tray: Installation

---
**The package has not been released on PyPI yet. As of now, only the local installation is available.**

---

The package can simply be installed using `pip` by running

    pip3 install deepsig

Another option is to clone the repository and install the package locally:

    git clone https://github.com/Kaleidophon/deep-significance.git
    cd deep-significance
    pip3 install -e .

**Warning**: Installed like this, imports will fail when the clones repository is moved.

## :bookmark: Examples

---
**tl;dr**: Use `aso()` to compare scores for two models. If the returned `eps_min < 0.5`, A is better than B.

---

In the following, I will lay out three scenarios that describe common use cases for ML practitioners and how to apply 
the methods implemented in this package accordingly. For an introduction into statistical hypothesis testing, please
refer to resources such as [this blog post](https://machinelearningmastery.com/statistical-hypothesis-tests/) for a general
overview or [Dror et al. (2018)](https://www.aclweb.org/anthology/P18-1128.pdf) for a NLP-specific point of view. 

In general, in statistical significance testing, we usually compare two algorithms <img src="svgs/53d147e7f3fe6e47ee05b88b166bd3f6.svg?invert_in_darkmode" align=middle width=12.32879834999999pt height=22.465723500000017pt/> and <img src="svgs/61e84f854bc6258d4108d08d4c4a0852.svg?invert_in_darkmode" align=middle width=13.29340979999999pt height=22.465723500000017pt/> on a dataset <img src="svgs/cbfb1b2a33b28eab8a3e59464768e810.svg?invert_in_darkmode" align=middle width=14.908688849999992pt height=22.465723500000017pt/> using 
some evaluation metric <img src="svgs/b5eaea000e06d5cf2e882f8fdbc71e36.svg?invert_in_darkmode" align=middle width=19.740822749999992pt height=22.465723500000017pt/>. The difference between the two algorithms on the data is then defined as 

<p align="center"><img src="svgs/9540dc879d2ecaa7cb245871b24f4e5d.svg?invert_in_darkmode" align=middle width=212.73480854999997pt height=16.438356pt/></p>

where <img src="svgs/6dea53e880ae565b82d6b4a6148a0012.svg?invert_in_darkmode" align=middle width=35.622171749999985pt height=24.65753399999998pt/> is our test statistic. We then test the following **null hypothesis**:

<p align="center"><img src="svgs/1d210dbbb93bbdc5a632b9443059499d.svg?invert_in_darkmode" align=middle width=100.49629589999999pt height=16.438356pt/></p>

Thus, we assume our algorithm A to be equally as good or worse than algorithm B and reject the null hypothesis if A 
is better than B (what we actually would like to see). Most statistical significance tests operate using 
*p-values*, which define the probability that under the null-hypothesis, the true difference <img src="svgs/6dea53e880ae565b82d6b4a6148a0012.svg?invert_in_darkmode" align=middle width=35.622171749999985pt height=24.65753399999998pt/> is larger than or
equal to the observed difference <img src="svgs/ecdae90a73f512871267f358443bd563.svg?invert_in_darkmode" align=middle width=26.32659479999999pt height=22.831056599999986pt/> (that is, for a one-sided test):

<p align="center"><img src="svgs/6d2735c4e335ec03c8b45736da4531a3.svg?invert_in_darkmode" align=middle width=135.91559685pt height=16.438356pt/></p>

Intuitively, the p-value is expressing: **How likely is it that the observed difference is up to what we expected, given that A is 
not better than B?** If this probability is high, it means that we're likely to see A is not better than B. If the 
probability is low, that means that <img src="svgs/94ea44af3034479a1ba3f2f655bcec39.svg?invert_in_darkmode" align=middle width=26.32659479999999pt height=22.831056599999986pt/> is likely *larger* than <img src="svgs/6dea53e880ae565b82d6b4a6148a0012.svg?invert_in_darkmode" align=middle width=35.622171749999985pt height=24.65753399999998pt/> - indicating 
that the null hypothesis might be wrong and that A is indeed better than B. 

To decide when we think A to be better than B, we set a threshold that will determine when we reject the null hypothesis, this is called the significance level <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> and it is often set to be 0.05.


### Intermezzo: Almost Stochastic Order - a better significance test for Deep Neural Networks

Deep neural networks are highly non-linear models, having their performance highly dependent on hyperparameters, random 
seeds and other (stochastic) factors. Therefore, comparing the means of two models across several runs might not be 
enough to decide if a model A is better than B. In fact, **even aggregating more statistics like standard deviation, minimum
or maximum might not be enough** to make a decision. For this reason, Dror et al. (2019) introduced *Almost Stochastic 
Order* (ASO), a test to compare two score distributions. 

It builds on the concept of *stochastic order*: We can compare two distribution and declare one as *stochastically dominant*
by comparing their cumulative distribution functions: 

![](img/so.png)

If the CDF of A is lower than B for every <img src="svgs/332cc365a4987aacce0ead01b8bdcc0b.svg?invert_in_darkmode" align=middle width=9.39498779999999pt height=14.15524440000002pt/>, we know the corresponding to algorithm A scores higher. However, in practice
these cases are rarely so clear-cut (imagine e.g. two normal distributions with the same mean but different variances).
For this reason, Dror et al. (2019) consider the notion of *almost stochastic dominance* by quantifying the extent to 
which stochastic order is being violated (red area):

![](img/aso.png)

ASO returns a value <img src="svgs/70bcb72c245ba47b6fc7439da91ec6fc.svg?invert_in_darkmode" align=middle width=28.45332764999999pt height=14.15524440000002pt/>, which expresses the amount of violation of stochastic order. If 
<img src="svgs/dabed7f05cf133d9eb92631d564a96a8.svg?invert_in_darkmode" align=middle width=72.19750559999999pt height=21.18721440000001pt/>, A is stochastically dominant over B in more cases than vice versa, then the corresponding algorithm can be declared as 
superior. We can also interpret <img src="svgs/70bcb72c245ba47b6fc7439da91ec6fc.svg?invert_in_darkmode" align=middle width=28.45332764999999pt height=14.15524440000002pt/> as a *confidence score*. The lower it is, the more sure we can be 
that A is better than B. Note: **ASO does not compute p-values.** Instead, the null hypothesis formulated as 

<p align="center"><img src="svgs/69c5ac8ce10d0dbd0c2b915aaf0472c1.svg?invert_in_darkmode" align=middle width=106.93478895pt height=13.698590399999999pt/></p>

If we want to be more confident about the result of ASO, we can also set the rejection threshold to be lower than 0.5.
Furthermore, the significance level <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is determined as an input argument when running ASO and actively influence 
the resulting <img src="svgs/70bcb72c245ba47b6fc7439da91ec6fc.svg?invert_in_darkmode" align=middle width=28.45332764999999pt height=14.15524440000002pt/>.


### Scenario 1 - Comparing multiple runs of two models 

In the simplest scenario, we have retrieved a set of scores from a model A and a baseline B on a dataset, stemming from 
various model runs with different seeds. We can now simply apply the ASO test:

```python
import numpy as np
from deepsig import aso

# Simulate scores
N = 5  # Number of random seeds
scores_a = np.random.normal(loc=0.9, scale=0.8, size=N)
scores_b =  np.random.normal(loc=0, scale=1, size=N)

min_eps = aso(scores_a, scores_b)  # min_eps = 0.0, so A is better
```

ASO **does not make any assumptions about the distributions of the scores**. 
This means that we can apply it to any kind of test metric. The more scores of model runs is supplied, the more reliable 
the test becomes. 

`aso()` runs with `build_quantile="fast"` by default. This runs the test quicker, but trades speed of accuracy. Thus, 
when obtaining <img src="svgs/70bcb72c245ba47b6fc7439da91ec6fc.svg?invert_in_darkmode" align=middle width=28.45332764999999pt height=14.15524440000002pt/> scores that are not very clear-cut (as indicated by a warning), please run 
the function again with `build_quantile="exact"`. 

### Scenario 2 - Comparing multiple runs across datasets

When comparing models across datasets, we formulate on null hypothesis per dataset. However, we have to make sure not to 
fall prey to the [multiple comparisons problem](https://en.wikipedia.org/wiki/Multiple_comparisons_problem): In short, 
the more comparisons between A and B we are conducting, the more likely gets is to reject a null-hypothesis accidentally.
That is why we have to adjust our significance threshold <img src="svgs/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode" align=middle width=10.57650494999999pt height=14.15524440000002pt/> accordingly by dividing it by the number of comparisons, 
which corresponds to the Bonferroni correction (Bonferroni et al., 1936):

```python 
import numpy as np
from deepsig import aso 

# Simulate scores for three datasets
M = 3  # Number of datasets
N = 5  # Number of random seeds
scores_a = [np.random.normal(loc=0.3, scale=0.8, size=N) for _ in range(M)]
scores_b = [np.random.normal(loc=0, scale=1, size=N) for _ in range(M)]

# epsilon_min values with Bonferroni correction 
eps_min = [aso(a, b, confidence_level=0.05 / M) for a, b in zip(scores_a, scores_b)]
# eps_min = [0.1565800030782686, 1, 0.0]
```

---
**Note**: When using another significance test like bootstrap (`from deepsig import bootstrap_test`), permutation-randomization
(`from deepsig import permutation test`) or e.g. a t-test (not implemented here), you can use the p-value correction 
implemented in `deep-significance` by `from deepsig import bonferroni_correcton`. `correct_p_value()` is **not** 
compatible with `aso()`.

---

### Scenario 3 - Comparing sample-level scores

In previous examples, we have assumed that we compare two algorithms A and B based on their performance per run, i.e. 
we run each algorithm once per random seed and obtain exactly one score on our test set. In some cases however, 
we would like to compare two algorithms based on scores **for every point in the test set**. If we only use one seed
per model, then this case is equivalent to scenario 1. But what if we also want to use multiple seeds per model?

In this scenario, we can do pair-wise comparisons of the score distributions between A and B and use the Bonferroni 
correction accordingly:

```python 
from itertools import product 

import numpy as np
from deepsig import aso 

# Simulate scores for three datasets
M = 40   # Number of data points
N = 3  # Number of random seeds
scores_a = [np.random.normal(loc=0.3, scale=0.8, size=M) for _ in range(N)]
scores_b = [np.random.normal(loc=0, scale=1, size=M) for _ in range(N)]
pairs = list(product(scores_a, scores_b))

# epsilon_min values with Bonferroni correction 
eps_min = [aso(a, b, confidence_level=0.05 / len(pairs)) for a, b in pairs]
```

### Scenario 4 - Comparing more than two models 

Similarly, when comparing multiple models (now again on a per-seed basis), we can use a similar approach like in the 
previous example. For instance, for three models, we can create a <img src="svgs/9f2b6b0a7f3d99fd3f396a1515926eb3.svg?invert_in_darkmode" align=middle width=36.52961069999999pt height=21.18721440000001pt/> matrix and fill the entries 
with the corresponding <img src="svgs/70bcb72c245ba47b6fc7439da91ec6fc.svg?invert_in_darkmode" align=middle width=28.45332764999999pt height=14.15524440000002pt/> values. The diagonal will naturally always be 1, but we can also restrict 
ourself to only filling out one half of the matrix by making use of the following property of ASO:

<p align="center"><img src="svgs/19a0661139e4b089cac41a8a27b738e7.svg?invert_in_darkmode" align=middle width=245.9131653pt height=16.438356pt/></p>

---
**Note**: This property is known to hold more for `build_quantile="fast"` (deviation of < 0.01) and less for 
`build_quantile="exact"` (deviation of < 0.2). This is because the quantile function is built based on resampled scores
during bootstrap iterations for the latter case, leading to more randomness when fewer scores for A and B are available, 
but also a tighter bound for <img src="svgs/70bcb72c245ba47b6fc7439da91ec6fc.svg?invert_in_darkmode" align=middle width=28.45332764999999pt height=14.15524440000002pt/> overall.

---

The corresponding code can then look something like this:

```python 
import numpy as np 
from deepsig import aso 
 
N = 5  # Number of random seeds
M = 3  # Number of different models / algorithms
num_comparisons = M * (M - 1) / 2
eps_min = np.eye(M)  # M x M matrix with ones on diagonal

scores_a = [np.random.normal(loc=0.3, scale=0.8, size=N) for _ in range(M)]
scores_b = [np.random.normal(loc=0, scale=1, size=N) for _ in range(M)]

for i in range(M):
  for j in range(i + 1, M):
    
    e_min = aso(scores_a[i], scores_b[j], confidence_level=0.05 / num_comparisons)
    eps_min[i, j] = e_min
    eps_min[j, i] = 1 - e_min
    
# eps_min =
# [[1.        1.         0.96926677]
# [0.         1.         0.71251641]
# [0.03073323 0.28748359 1.        ]]
```

### General recommendations & other notes

* Naturally, the CDFs built from `scores_a` and `scores_b` can only be approximations of the true distributions. Therefore,
as many scores as possible should be collected, especially if the variance between runs is high. If only one run is available,
  comparing sample-wise score distributions like in scenario 3 can be an option, but comparing multiple runs will 
  **always** be preferable.

* `num_samples` and `num_bootstrap_iterations` can be reduced to increase the speed of `aso()`. However, this is not 
recommended as the result of the test will also become less accurate. Technically, <img src="svgs/70bcb72c245ba47b6fc7439da91ec6fc.svg?invert_in_darkmode" align=middle width=28.45332764999999pt height=14.15524440000002pt/> is a upper bound
  that becomes tighter with the number of samples and bootstrap iterations (del Barrio et al., 2017). 
  
* Bootstrap and permutation-randomization are all non-parametric tests, i.e. they don't make any assumptions about 
the distribution of our test metric. Nevertheless, they differ in their *statistical power*, which is defined as the probability
  that the null hypothesis is being rejected given that there is a difference between A and B. In other words, the more powerful 
  a test, the less conservative it is and the more it is able to pick up on smaller difference between A and B. Therefore, 
  if the distribution is known or found out why normality tests (like e.g. Anderson-Darling or Shapiro-Wilk), something like 
  a parametric test like Student's or Welch's t-test is preferable to bootstrap or permutation-randomization. However, 
  because these test are in turn less applicable in a Deep Learning setting due to the reasons elaborated on in 
  [Why?](#interrobang-why), ASO is still a better choice.

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
The inline latex equations were rendered using [readme2latex](https://github.com/leegao/readme2tex).

### :books: Bibliography

Del Barrio, Eustasio, Juan A. Cuesta-Albertos, and Carlos Matrán. "An optimal transportation approach for assessing almost stochastic order." The Mathematics of the Uncertain. Springer, Cham, 2018. 33-44.

Bonferroni, Carlo. "Teoria statistica delle classi e calcolo delle probabilita." Pubblicazioni del R Istituto Superiore di Scienze Economiche e Commericiali di Firenze 8 (1936): 3-62.

Borji, Ali. "Negative results in computer vision: A perspective." Image and Vision Computing 69 (2018): 1-8.

Dror, Rotem, et al. "The hitchhiker’s guide to testing statistical significance in natural language processing." Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2018.

Dror, Rotem, Shlomov, Segev, and Reichart, Roi. "Deep dominance-how to properly compare deep neural models." Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. 2019.

Efron, Bradley, and Robert J. Tibshirani. "An introduction to the bootstrap." CRC press, 1994.

Henderson, Peter, et al. "Deep reinforcement learning that matters." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 32. No. 1. 2018.

Hao Li, Zheng Xu, Gavin Taylor, Christoph Studer, Tom Goldstein. "Visualizing the Loss Landscape of Neural Nets." NeurIPS 2018: 6391-6401

Narang, Sharan, et al. "Do Transformer Modifications Transfer Across Implementations and Applications?." arXiv preprint arXiv:2102.11972 (2021).

Noreen, Eric W. "Computer intensive methodsfor hypothesis testing: An introduction." Wiley,New York (1989).
