# Bi-level-optim

## Preface
This code was executed in a virtual python environment using Python 3.9.
This Code belongs to the AutoML Conf 2024 Paper titeled "Bi-Level One-Shot Architecture Search for Probabilistic Time Series Forecasting".
This Code contains everything necessary to run experiments on the "Exchange" as well as "Power" dataset referenced in the paper.
To obtain the other dataset follow these steps: <br>
M4 <br>
  * Download the data from https://www.kaggle.com/datasets/yogesh94/m4-forecasting-competition-dataset/data <br>
Solar, Wiki <br>
  * Use get_dataset function of this library https://ts.gluon.ai/stable/tutorials/forecasting/quick_start_tutorial.html <br>
  * Call this function either with "solar_nips" or "wiki2000_nips" <br>
<br>
All preprocessing steps and datasplits are automaticly conducted internally and identical to the procedure used for the paper <br>
Store the .csv files (in case of M4) or a .pkl file of the data (in case of Solar, and Wiki) at the following places: <br>

  * Bi-level-optim/ArchSearch/darts/res/ <br>
  * Bi-level-optim/ArchSearch/res/ <br>
  * Bi-level-optim/PWN/res/ <br>
  
Call the corresponding folders either "M4", "solar_data", or "wiki_data" <br>

## Running the architecture search
To run the architecture search use the *train_search.py* file in Bi-level-optim/ArchSearch/darts/ <br>
The standard experiment to run is a macro-architecture search for "Exchange". <br>
If you like to run a different experiment setup use the options stated in lines 126 to 141. <br>
Please set only one dataset flag to true at a time.
```
cd Bi-level-optim/ArchSearch/darts/
python train_search.py
```

## Training PWN
To train PWN, either standard or with an optimized architecture use the *training.py* file in Bi-level-optim/PWN/ <br>
The standard experiment to run is the training of an optimized architecture for "Exchange". <br>
```
cd Bi-level-optim/PWN/
python training.py
```

> NOTE: If you want to train on M4, please use `python trainin_m4.py`.

In line 26 and 58 you can choose wether or not to use an optimized architecture <br>
If you want to run an experiment with the larger vanilla model mentioned in the paper set the <br>
aforementioned options to False and set *config.rnn_layer_config.n_layers* in line 37 to 12. <br>
To other experiments different experiments define your own at line 105 to 114 (multiple at oncee possible). <br>
Possible selection options are:<br>
  * ReadX -> Choose the appropiate read function for your dataset <br>
  * PWN or PWNEM -> PWN uses CWSPN and PWNEM uses WEin as the PC <br>
  * config_c -> Replace with config_w if you use PWNEM <br>
  * context_timespan, prediction_timespan, timespan_step -> replace according to dataset <br>
  * use_transformer -> choose wether to use STrans <br>
  * smape_target -> Set to true if SMAPE is your metric (only for M4) <br>
Important Notes:
  * batchsize and #epochs can be changed in line 152 <br>
  * comment out all instances of config.window_size and config.fft_compression except for the one related to your dataset (line 80 to 99) <br>

To use a new micro architecture for the SRNN and CWSPN copy the results of the architecture search (the arch. weight arrays) to *pwn.py* in  <br>
Bi-level-optim/PWN/model/ in line 21 (SRNN) or line 25 (CWSPN). If you use PWNEM you can do the same for a searched SRNN arch in file *pwn_em.py*  <br>
in the same folder.

## Citation
```
@inproceedings{
seng2024bilevel,
title={Bi-Level One-Shot Architecture Search for Probabilistic Time Series Forecasting},
author={Jonas Seng and Fabian Kalter and Zhongjie Yu and Fabrizio Ventola and Kristian Kersting},
booktitle={AutoML 2024 Methods Track},
year={2024},
url={https://openreview.net/forum?id=AaPhnfFQYn}
}
```