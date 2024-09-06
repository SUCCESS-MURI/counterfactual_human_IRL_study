# User Study on Machine Teaching

This repository contains code for the user study described in the following [paper](https://arxiv.org/pdf/2203.01855): 

- Michael S. Lee, Henny Admoni, Reid Simmons **Reasoning about Counterfactuals to Improve Human Inverse Reinforcement Learning**, IROS, 2022,

which introduces a method for teaching robot decision making to humans through demonstrations of the robot's key decisions in a domain. 

In particular, we observe that an informative demonstration is one that differs strongly from the human's expectations of what the robot will do given their current understanding of the robot’s decision making, and we consider human beliefs and counterfactuals over robot behavior when selecting demonstrations accordingly.

This user study tests whether our proposed method for selecting demonstrations with consideration for human beliefs and counterfactuals over robot behavior improves humans' ability to predict robot behavior in new situations over a baseline that selects demonstrations without such human modeling.

Please find additional information and follow-on work at https://symikelee.github.io/.

## Running the user study

Run ``python herokuapp.py`` to run the user study locally. You must have [psiturk](https://psiturk.org/) installed, and the study must be completed on a laptop or desktop on one of the following browsers: Chrome, Firefox, or Safari. Please note that this code has only been tested with Python 3.8 and Psiturk version 3.1.0 (which can be configured using a package manager such as [conda](https://docs.conda.io/en/latest/)).

Videos of teaching demonstrations reside in `static/vid/{baseline, counterfactual_only, feature_only, proposed}`. You may navigate to these directories and directly view the videos without running the user study.

## Data analysis

`dfs_processed_masked.pickle` contains the user study data, which can be analyzed using `data_analysis_S22.py`. 
