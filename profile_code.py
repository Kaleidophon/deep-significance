# TODO: Delete this

import numpy as np
from deepsig.aso import aso
import cProfile

pr = cProfile.Profile()
samples_normal1 = np.random.normal(size=1000)  # Scores for algorithm A
samples_normal2 = np.random.normal(size=1000)  # Scores for algorithm B
pr.enable()
vr, _ = aso(
    samples_normal1,
    samples_normal2,
    num_bootstrap_iterations=1000,
    num_samples_a=1000,
    num_samples_b=1000,
)
pr.disable()
pr.print_stats(sort="calls")
