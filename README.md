# PANDA

Power efficiency is a critical design objective in modern microprocessor design. To evaluate the impact of architectural-level design decisions, an accurate yet efficient architecture-level power model is desired. However, widely adopted data-independent analytical power models like McPAT and Wattch have been criticized for their unreliable accuracy. While some machine learning (ML) methods have been proposed for architecture-level power modeling, they rely on sufficient known designs for training and perform poorly when the number of available designs is limited, which is typically the case in realistic scenarios. PANDA is an architecture-level power evaluation method by unifying analytical and machine learning solutions. We propose PANDA, an innovative architecture-level solution that combines the advantages of analytical and ML power models. It achieves unprecedented high accuracy on unknown new designs even when there are very limited designs for training, which is a common challenge in practice.


## Introduction
The modeling flow of McPAT-Calib can be divided into two parts, the focus is on PANDA Power Evaluation:
1) Microarchitecture Simulation: Use the microarchitecture simulator (gem5) to complete the simulation of the given BOOM microarchitecture configuration and benchmark.
2) PANDA Power Evaluation: Using the configuration parameters and events generated by the microarchitecture simulator to predict the power consumption.

## Quick Start
We prepared a example dataset in "example_data", with feature_set and label_set. So you can just run the PANDA power model, the result will be visulized as figures in "result_figure"
```
cd power_model
python PANDA.py
```

## Generate Dataset by Yourself
We also provide the flow to generate dataset in this project, so you can easily use PANDA on your own data. This can be divided into two parts:
1) Generate Feature: Use gem5 to generate related events. What's more, to help users to compare PANDA to other works, we also provide the flow to run McPAT although PANDA doesn't need the output of McPAT.
```
cd arch_sim_flow/
python 1_run_gem5.py
python 2_run_mcpat.py
python 3_microarchitecture_data_processing.py
```
2) Generate Label: Use VLSI flow to generate ground truth.
```
cd vlsi_flow/
python power_analysis_data_processing.py
```
