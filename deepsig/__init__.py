# Make main functions accessible on a package level for cleaner imports
from deepsig.aso import aso, multi_aso
from deepsig.bootstrap import bootstrap_test
from deepsig.correction import bonferroni_correction
from deepsig.permutation import permutation_test
from deepsig.sample_size import aso_uncertainty_reduction, bootstrap_power_analysis

__version__ = "1.2.3"
__author__ = "Dennis Ulmer"
