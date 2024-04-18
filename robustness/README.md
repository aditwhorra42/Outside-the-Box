## Introduction
The main contribution of this work is to analyse the general robustness of monitoring frameworks to targeted adversarial examples. In order to perform this analysis, there is need for a metric that reliably captures the robustness of a monitor and expresses it as a standardised score between 0 and 1. Such a metric will establish a basis for quantifiable robustness for monitors and will also provide a way to compare the robustness between different monitors. Therefore, this work introduces a new metric, specifically designed for monitoring frameworks, that quantifies the robustness of a monitor as a normalised score between 0 and 1. Moreover, as part of the analysis, this work also presents a comparative study of the robustness of monitors with varying levels of perturbation and number of known classes.

The results can be seen by unzipping the final_experiments.zip file.


## How to run the code
The three trained monitors are stored in the trained_monitors folder. To generate the adversarial exampples, edit the parameters in the main methid in `run_experiment_adversarial.py` and run it. Once the results are saved in the save_path provided, edit the parameters in `calculate_robustness_score.py` and run it to get the robustness scores.
