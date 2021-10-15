import numpy as np
from deepsig import multi_aso

N = 5  # Number of random seeds
M = 3  # Number of different models / algorithms

# Same setup as above, but use a dict for scores
my_models_scores = {
    f"model {i + 1}": np.random.normal(loc=loc, scale=0.8, size=N)
    for i, loc in enumerate(np.arange(0.1, 0.1 + 0.05 * M, step=0.05))
}

# my_model_scores = {
#   "model 1": array([...]),
#   "model 2": array([...]),
#   ...
# }

eps_min = multi_aso(my_models_scores, confidence_level=0.05, return_df=True)

# This is now a DataFrame!
# eps_min =
#           model 1   model 2  model 3
# model 1       1.0       1.0      1.0
# model 2       0.0       1.0      1.0
# model 3       1.0       0.0      1.0
