# deep-significance: Easy Significance Testing for Deep Neural Networks

@TODO: Shields

**Warning: This project is still under development. Code might be erroneous and breaking changes be introduced without 
warning.**

### :interrobang: Why?

Although the field of Machine Learning and Deep Learning has undergone spectacular growth in the recent decade (@CITE),
a large portion of experimental evidence is not supported by statistical significance tests (@CITE). Instead, 
conclusions are often drawn based on single performance scores. This is problematic: Neural network performance depends
on many factors: The specific hyperparameters that were found, stochastic factors like Dropout masks (@TODO: more 
factors). 

To help mitigate this problem, this package supplies fully-tested reimplementations of useful functions for significance
testing:
* Non-parametic test such as Almost Stochastic Order (Dror et al., 2019), bootstrap and permutation-randomization.
* p-value corrections methods such as Bonferroni and Fisher. 

For examples about the usage, consult the documentation here (@TODO: Add link to docs) or the scenarios 
in the section [Examples](#examples).

## :inbox_tray: Installation

(**The package has not been released on pip yet**)

The package can simply be installed using `pip` by running

    pip3 install deepsig

Alternatively, the repository can also be installed by cloning the repo first:

    git clone https://github.com/Kaleidophon/deep-significance.git
    cd deep-significance 
    pip3 install -e .

### :bulb: A short and gentle introduction to significance testing (for DNNs)

@TODO: Significant testing basic idea
@TODO: Multiple comparisons
@TODO: Significance testing for neural networks

## :bookmark: Examples

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

### :medal_sports: Credit

@TODO

### :books: Bibliography

@TODO
