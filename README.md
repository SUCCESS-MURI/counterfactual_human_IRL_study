# User Study on Machine Teaching

Run ``python herokuapp.py`` to run the user study locally. You must have [psiturk](https://psiturk.org/) installed, and the study must be completed on a laptop or desktop on one of the following browsers: Chrome, Firefox, or Safari. Please note that this code has only been tested with Python 3.8 and Psiturk version 3.1.0 (which can be configured using a package manager such as [conda](https://docs.conda.io/en/latest/)).

dfs_processed_masked.pickle contains the user study data, which can be analyzed using data_analysis_S22.py. 

Videos of teaching demonstrations reside in static/vid/{baseline, counterfactual_only, feature_only, proposed}. You may navigate to these directories and directly view the videos without running the user study.

This repository contains raw code that has not gone extensive cleanup, which will be done upon acceptance of the paper.