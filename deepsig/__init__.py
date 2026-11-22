# Make main functions accessible on a package level for cleaner imports
from deepsig.aso import aso, multi_aso
from deepsig.bootstrap import bootstrap_test
from deepsig.correction import (
    bonferroni_correction,
    correct_p_values,
    holm_correction,
)
from deepsig.permutation import permutation_test
from deepsig.sample_size import aso_uncertainty_reduction, bootstrap_power_analysis

__version__ = "1.3.0"
__author__ = "Dennis Ulmer"
