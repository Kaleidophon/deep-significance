# Make main functions accessible on a package level for cleaner imports
from deepsig.aso import aso
from deepsig.bootstrap import bootstrap_test
from deepsig.correction import bonferroni_correction
from deepsig.permutation import permutation_test

__version__ = "1.0.2"
__author__ = "Dennis Ulmer"
