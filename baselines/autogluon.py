# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:42:35 2024

@author: Fabian
"""
import torch
import argparse
import torch.utils
import warnings
import pandas as pd
#from preprocessing import *
import json
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor
from rtpt import RTPT

device = 'cuda'

#warnings.filterwarnings("ignore", message=".*affe2")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument('--key', type=str, default='M4_Yearly', help='Identifire String for the dataset to use')
parser.add_argument('--experiment', type=int, default=0, help='What PWN combination should be used, 0-SRNN+CWSPN ; 1-SRNN+WEin ; 2-STrans+CWWSPN ; 3-STrans+WEin')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')

args = parser.parse_args()

def main(rand):

  f = open('params.json')
  experiment_config = json.load(f)

  numEpochs = experiment_config[args.key]['numEpochs']
  batchsize = experiment_config[args.key]['batchsize']
  fftWinSize = experiment_config[args.key]['fftWinSize']
  fftComp = experiment_config[args.key]['fftComp']
  pwnLR = experiment_config[args.key]['pwnLR']
  contexttime = experiment_config[args.key]['contexttime']
  predictiontime = experiment_config[args.key]['predictiontime']

  torch.manual_seed(rand)

  if args.key == 'M4_Yearly':
    df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_yearly/test.csv")
    path = 'yearly'
    evm="SMAPE"
  elif args.key == 'M4_Quarterly':
    df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_quarterly/test.csv")
    path = 'quarterly'
    evm = "SMAPE"
  elif args.key == 'M4_Monthly':
    df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_monthly/test.csv")
    path = 'monthly'
    evm = "SMAPE"
  elif args.key == 'M4_Weekly':
    df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_weekly/test.csv")
    path = 'weekly'
    evm = "SMAPE"
  elif args.key == 'M4_Daily':
    df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_daily/test.csv")
    path = 'daily'
    evm = "SMAPE"
  elif args.key == 'M4_Hourly':
    df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/test.csv")
    path = 'hourly'
    evm = "SMAPE"
  #elif args.key == 'Power':
  #  data_source = ReadPowerPKL()
  #  evm = "MSE"
  #elif args.key == 'Exchange':
  #  data_source = ReadExchangePKL()
  #  evm = "MSE"
  #elif args.key == 'Wiki':
  #  data_source = ReadWikipediaPKL()
  #  evm = "MSE"
  #elif args.key == 'Solar':
  #  data_source = ReadSolarPKL()
  #  evm = "MSE"

  df = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/timeseries/m4_hourly/train.csv")

  data = TimeSeriesDataFrame.from_data_frame(
    df,
    id_column="item_id",
    timestamp_column="timestamp"
  )
  pred_length = predictiontime

  train_data, test_data = data.train_test_split(pred_length)

  predictor = TimeSeriesPredictor(
    prediction_length=pred_length,
    path=f"autogluon-m4-{path}",
    target="target",
    eval_metric="SMAPE",
  )

  time_limit = 60*60*6 # 6 hours for search to ensure fair comparison

  predictor.fit(
    train_data,
    presets="best_quality",
    time_limit=time_limit,
    random_seed=rand
  )

  scores = predictor.evaluate(test_data, metrics='SMAPE')
  log_file = f'autogluon-m4-{path}/{args.key}_{args.experiment}_{rand}.json'
  with open(log_file, 'w') as json_file:
    json.dump(scores, json_file, indent=4)


if __name__ == '__main__':
  torch.autograd.set_detect_anomaly(True)
  seeds = list(range(10))
  rt = RTPT('JS', 'PWN_baseline', 10)
  rt.start()
  for s in seeds:
    main(rand = s)
    rt.step()
