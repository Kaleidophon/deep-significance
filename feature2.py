import numpy as np
from deepsig import multi_aso

N = 5  # Number of random seeds
M = 3  # Number of different models / algorithms

# Simulate different model scores by sampling from normal distributions with increasing means
# Here, we will sample from N(0.1, 0.8), N(0.15, 0.8), N(0.2, 0.8)
my_models_scores = [
    np.random.normal(loc=loc, scale=0.8, size=N)
    for loc in np.arange(0.1, 0.1 + 0.05 * M, step=0.05)
]

eps_min = multi_aso(my_models_scores, confidence_level=0.05)

# eps_min =
# array([[1., 1., 1.],
#        [0., 1., 1.],
#        [0., 0., 1.]])
