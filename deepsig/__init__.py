# Make main functions accessible on a package level for cleaner imports
from deepsig.aso import aso, multi_aso
from deepsig.bootstrap import bootstrap_test
from deepsig.correction import bonferroni_correction
from deepsig.permutation import permutation_test

__version__ = "1.1.0"
__author__ = "Dennis Ulmer"
