# deep-significance - Paper README

This README quickly summarizes the usage and content of the code used in the companion paper for the package.
The paper adds additional software requirements to those of the package, which can be installed via

    pip3 install -r paper_requirements.txt

Afterwards, simply run `python3 test_comparison.py` to reproduce figures 2, 4 (a) and 5. Run 
`python3 test_aso_params.py` to reproduce figures 4 (b) + (c). In case you don't want to re-run all the scripts, all 
results are stores in the `img/` folder, including plots and the raw data in Python's `pickle` format.

The code for the demo used in section 6.1 is given in `deep-significance demo.ipynb`.