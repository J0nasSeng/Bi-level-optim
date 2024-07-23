# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:42:35 2024

@author: Fabian
"""

import os
import sys
import time
import glob
import numpy as np
import torch
import darts.utils as utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import warnings
import copy

from darts.darts_cnn.model_search import Network as CWSPNModelSearch
from darts.darts_rnn.model_search import RNNModelSearch
#from model.pwn_em_for_darts import PWNEM
#from model.pwn_for_darts import PWN
from model.pwn_em_v2_darts import PWNEM
from model.pwn_v2_darts import PWN
from model.transformer.transformer_config import TransformerConfig
from model.spectral_rnn import SpectralRNN, SpectralRNNConfig
from torch.autograd import Variable
from darts.model_search import Network
from darts.architect import Architect
from model.cwspn import CWSPN, CWSPNConfig
from model.wein import WEin, WEinConfig
from model.wein.EinsumNetwork.ExponentialFamilyArray import NormalArray, MultivariateNormalArray, BinomialArray
from data_source import BasicSelect, Mackey, ReadPowerPKL, ReadM4, ReadExchangePKL, ReadWikipediaPKL, ReadSolarPKL
from preprocessing import *
from torch.utils.data import TensorDataset, DataLoader
from data_source import M4Dataset
from torch.utils.data import random_split
import json
from rtpt import RTPT

device = 'cuda'



#warnings.filterwarnings("ignore", message=".*affe2")
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.01, help='init learning rate')#0.01 standard #0.001 for exchange
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=2, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=13, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=1e-4 , help='learning rate for arch encoding') #3e-4 standard
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')

parser.add_argument('--key', type=str, default='M4_Yearly', help='Identifire String for the dataset to use')
#parser.add_argument('--experiment', type=int, default=0, help='What PWN combination should be used, 0-SRNN+CWSPN ; 1-SRNN+WEin ; 2-STrans+CWWSPN ; 3-STrans+WEin')

args = parser.parse_args()

device = device + f':{args.gpu}'


def main(rand):

  save_file = 'search-{}-{}-{}'.format(args.save, args.key, rand)
  utils.create_exp_dir(save_file, scripts_to_save=glob.glob('*.py'))

  log_format = '%(asctime)s %(message)s'
  logging.basicConfig(stream=sys.stdout, level=logging.INFO,
      format=log_format, datefmt='%m/%d %I:%M:%S %p')
  fh = logging.FileHandler(os.path.join(save_file, 'log.txt'))
  fh.setFormatter(logging.Formatter(log_format))
  logging.getLogger().addHandler(fh)

# Set configs for PWN components #######################################################################################
  np.random.seed(rand)#np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(rand)#torch.manual_seed(args.seed)
  cudnn.enabled = True
  torch.cuda.manual_seed(rand)#torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)
  config = SpectralRNNConfig()
# config.normalize_fft = True
  config.use_add_linear = False
  config.rnn_layer_config.use_gated = True
  config.rnn_layer_config.use_cg_cell = False
  config.rnn_layer_config.use_residual = True
  config.rnn_layer_config.learn_hidden_init = False
  config.rnn_layer_config.use_linear_projection = True
  config.rnn_layer_config.dropout = 0.1
  config.window_size = 96  # m4_settings[m4_key]['window_size']
  config.fft_compression = 4  # m4_settings[m4_key]['fft_compression']
  config.hidden_dim = 128
  config.rnn_layer_config.n_layers = 2
  config.use_cached_predictions = False

  config_t = TransformerConfig(normalize_fft=True, window_size=config.window_size,
                              fft_compression=config.fft_compression)

  config_w = WEinConfig()
  config_w.exponential_family = NormalArray  # NormalArray #MultivariateNormalArray
  config_w.window_level = False
  config_w.mpe_prediction = False
  config_w.structure = {'type': 'binary-trees', 'depth': 4, 'num_repetitions': 5}
  config_w.exponential_family_args = {'min_var': 1e-4, 'max_var': 4.}
  config_w.prepare_joint = False
  config_w.K = 2

  config_c = CWSPNConfig()
  config_c.num_gauss = 2
  config_c.num_sums = 4
  config_c.rg_splits = 8
  config_c.rg_split_recursion = 2
  config_c.gauss_min_sigma = 1e-4
  config_c.gauss_max_sigma = 1. * 4
  config_c.use_rationals = True

########################################################################################################################

  use_Power = False

  search_srnn = True
  compare_search_srnn_to_transformer = True
  search_cwspn = True
  compare_search_cwspn_to_wein = False
  turn_off_wein = False


###################################  Load  Data ########################################################################

  f = open('params.json')
  experiment_config = json.load(f)

  archepochs = experiment_config[args.key]['archepochs']
  batchsize = experiment_config[args.key]['batchsize']
  config.window_size = experiment_config[args.key]['fftWinSize']
  config.fft_compression = experiment_config[args.key]['fftComp']
  earlyStop = experiment_config[args.key]['EarlyStop']
  context_timespan = experiment_config[args.key]['contexttime']
  prediction_timespan = experiment_config[args.key]['predictiontime']
  timespan_step = experiment_config[args.key]['timestep']
  archLR = experiment_config[args.key]['archLR']

  args.arch_learning_rate = archLR
  args.batch_size = batchsize

  ######  Choose Read Data Function  #########################
  smape_use = False
  config_layer = {} 

  if args.key == 'M4_Yearly':
      #data_source = ReadM4(key='Yearly')
      smape_use = True
      hidden_size_trans = 128
      hiddeen_size_srnn = 64
      m4_key = args.key[3:].lower()
      dataset = M4Dataset(train_len=context_timespan,
                          test_len=prediction_timespan, subset=m4_key)
      value_dim = config.window_size // 2 + 1
      config_layer['input_size'] = value_dim // config.fft_compression
      config_layer['hidden_size'] = 64
      config_layer['fftcomp'] = config.fft_compression
      config_layer['windowsize'] = config.window_size

  elif args.key == 'M4_Quaterly':
      #data_source = ReadM4(key='Quarterly')
      smape_use = True
      hidden_size_trans = 128
      hiddeen_size_srnn = 64
      m4_key = args.key[3:].lower()
      dataset = M4Dataset(train_len=context_timespan,
                          test_len=prediction_timespan, subset=m4_key)
      value_dim = config.window_size // 2 + 1
      config_layer['input_size'] = value_dim // config.fft_compression
      config_layer['hidden_size'] = 64
      config_layer['fftcomp'] = config.fft_compression
      config_layer['windowsize'] = config.window_size

  elif args.key == 'M4_Monthly':
      #data_source = ReadM4(key='Monthly')
      smape_use = True
      hidden_size_trans = 128
      hiddeen_size_srnn = 64
      m4_key = args.key[3:].lower()
      dataset = M4Dataset(train_len=context_timespan,
                          test_len=prediction_timespan, subset=m4_key)
      value_dim = config.window_size // 2 + 1
      config_layer['input_size'] = value_dim // config.fft_compression
      config_layer['hidden_size'] = 64
      config_layer['fftcomp'] = config.fft_compression
      config_layer['windowsize'] = config.window_size

  elif args.key == 'M4_Weekly':
      #data_source = ReadM4(key='Weekly')
      smape_use = True
      hidden_size_trans = 128
      hiddeen_size_srnn = 64
      m4_key = args.key[3:].lower()
      dataset = M4Dataset(train_len=context_timespan,
                          test_len=prediction_timespan, subset=m4_key)
      value_dim = config.window_size // 2 + 1
      config_layer['input_size'] = value_dim // config.fft_compression
      config_layer['hidden_size'] = 64
      config_layer['fftcomp'] = config.fft_compression
      config_layer['windowsize'] = config.window_size

  elif args.key == 'M4_Daily':
      #data_source = ReadM4(key='Daily')
      smape_use = True
      hidden_size_trans = 128
      hiddeen_size_srnn = 64
      m4_key = args.key[3:].lower()
      dataset = M4Dataset(train_len=context_timespan,
                          test_len=prediction_timespan, subset=m4_key)
      value_dim = config.window_size // 2 + 1
      config_layer['input_size'] = value_dim // config.fft_compression
      config_layer['hidden_size'] = 64
      config_layer['fftcomp'] = config.fft_compression
      config_layer['windowsize'] = config.window_size

  elif args.key == 'M4_Hourly':
      #data_source = ReadM4(key='Hourly')
      smape_use = True
      hidden_size_trans = 128
      hiddeen_size_srnn = 64
      m4_key = args.key[3:].lower()
      dataset = M4Dataset(train_len=context_timespan,
                          test_len=prediction_timespan, subset=m4_key)
      value_dim = config.window_size // 2 + 1
      config_layer['input_size'] = value_dim // config.fft_compression
      config_layer['hidden_size'] = 64
      config_layer['fftcomp'] = config.fft_compression
      config_layer['windowsize'] = config.window_size

  elif args.key == 'Power':
      data_source = ReadPowerPKL()
      use_Power = True
  elif args.key == 'Exchange':
      data_source = ReadExchangePKL()
  elif args.key == 'Wiki':
      data_source = ReadWikipediaPKL()
  elif args.key == 'Solar':
      data_source = ReadSolarPKL()


  dataset.prepare()
  split = int(0.8 * len(dataset.train_data))
  train_data, test_data = random_split(dataset.train_data, [split, len(dataset.train_data) - split])

  train_loader = DataLoader(train_data, shuffle=True, batch_size=batchsize, drop_last=True)
  test_loader = DataLoader(test_data, shuffle=True, batch_size=batchsize, drop_last=True)

  #val_data = TensorDataset(x_test, y_test)
  #val_loader = DataLoader(test_data, shuffle=True, batch_size=batchsize, drop_last=True)

########################################################################################################################
  model_1 = PWN(hidden_size_trans, prediction_timespan, config.fft_compression,
                  config.window_size, 0.5, device, config_c, num_srnn_layers=2, train_spn_on_gt=False, train_spn_on_prediction=True,
                smape_target=smape_use, use_transformer=True, train_rnn_w_ll=False)

  model_2 = PWNEM(hiddeen_size_srnn, prediction_timespan, config.fft_compression,
                        config.window_size, 0.5, device, config_w, num_srnn_layers=2, train_spn_on_gt=False, train_spn_on_prediction=True,
                        train_rnn_w_ll=False, use_transformer=False, smape_target=smape_use)

  #this is not actually a training step, this is just a simple way to build the standard network
  #model_1 = model_1.prepareNets(train_loader)
  #model_2 = model_2.prepareNets(train_loader)
  model_1.prepareNets(train_loader)
  model_2.prepareNets(train_loader)
  srnn = model_2.srnn
  wein = model_2.westimator.net


  transformer = model_1.srnn
  cwspn_nn = model_1.westimator.weight_nn
  cwspn_spn = model_1  # model_1.westimator.spn

  search_stft = srnn.stft
############################# Params of Darts that are not needed but functions havn't been refactored yet #############
  emsize = 300
  nhid = 300
  nhidlast = 300
  ntokens = 10000
  dropout = 0
  dropouth = 0
  dropouti = 0
  dropoute = 0
  dropoutx = 0
########################################################################################################################



  in_seq_length = model_1.westimator.input_sizes[0] * (2 if model_1.westimator.use_stft else 1) # input sequence length into the WeightNN
  output_length = model_1.westimator.num_sum_params + model_1.westimator.num_leaf_params # combined length of sum and leaf params
  sum_params = model_1.westimator.num_sum_params
  cwspn_weight_nn_search = CWSPNModelSearch(in_seq_length, output_length, sum_params, layers=1, steps=4)

  srnn_search = RNNModelSearch(search_stft, config_layer, ntokens, emsize, nhid, nhidlast,
                             dropout, dropouth, dropoutx, dropouti, dropoute)

  transformer = model_1.srnn
  cwspn_nn = model_1.westimator.weight_nn
  cwspn_spn = model_1#model_1.westimator.spn

  spn_nn_modul_1 = cwspn_nn
  spn_nn_modul_2 = wein

  spn_modul_1 = cwspn_spn
  spn_modul_2 = []

  spectral_farcaster_modul_1 = model_1.srnn #Transformer
  spectral_farcaster_modul_2 = model_2.srnn #SRNN

  if search_srnn:
    if compare_search_srnn_to_transformer:
      spectral_farcaster_modul_2 = srnn_search
      #spectral_farcaster_modul_2 = []
    else:
      spectral_farcaster_modul_1 = model_2.srnn #SRNN
      spectral_farcaster_modul_2 = srnn_search

  if search_cwspn:

      spn_nn_modul_1 = cwspn_weight_nn_search

      if not compare_search_cwspn_to_wein:
        spn_nn_modul_2 = model_1.westimator.weight_nn
        spn_modul_2 = copy.copy(model_1)

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  model = Network(spectral_farcaster_modul_2, spectral_farcaster_modul_1, spn_modul_2,spn_nn_modul_2, spn_nn_modul_1, spn_modul_1,turn_off_wein, args.layers, criterion, search_srnn,search_cwspn,smape_target=smape_use)
  model = model.cuda()

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, args)

  rtpt = RTPT('JS', 'PWN_NAS', args.epochs)
  rtpt.start()

  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    #print(F.softmax(model.alphas_normal, dim=-1))
    #print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train_new(train_loader, test_loader, model, architect, criterion, optimizer, lr,model_1,smape_use,earlyStop)
    logging.info('train_acc %f', train_acc)

    if epoch < model.ll_weight_inc_dur and model.train_rnn_w_ll:
      if model.step_increase:
        current_ll_weight = 0
      else:
        model.current_ll_weight += model.ll_weight_increase
    elif model.train_rnn_w_ll:
      model.current_ll_weight = model.ll_weight

    if search_srnn:
      logging.info('SRNN weights %s', F.softmax(model.srnn_arch_weights, dim=-1).tolist())
    if search_cwspn:
      logging.info('CWSPN weights %s', F.softmax(model.cwspn_arch_weights, dim=-1).tolist())
    logging.info('Forcaster weights %s', F.softmax(model.alphas_normal, dim=-1).tolist())
    logging.info('SPN weights %s', F.softmax(model.alphas_reduce, dim=-1).tolist())

    utils.save(model, os.path.join(save_file, f'weights_{rand}.pt'))
    rtpt.step()

def train_new(train_loader,test_loader, model, architect, criterion, optimizer, lr,pwn_model,use_smape,earlyStop):
  objs = utils.AvgrageMeter()
  objs_spn = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()

  for step, (batch_x, batch_y) in enumerate(train_loader):

    model.train()
    n = batch_x.size(0)
    if step >= earlyStop:#90
      continue

    batch_x = Variable(batch_x, requires_grad=False).cuda()
    batch_y = Variable(batch_y, requires_grad=False).cuda(non_blocking=True)
    batch_x, batch_y = batch_x.to(torch.float32), batch_y.to(torch.float32)

    batch_westimator_x, batch_westimator_y = pwn_model.westimator.prepare_input(batch_x, batch_y)

    batch_westimator_x = Variable(batch_westimator_x, requires_grad=False).cuda()
    batch_westimator_y = Variable(batch_westimator_y, requires_grad=False).cuda(non_blocking=True)


    # get a random minibatch from the search queue with replacement
    batch_x_val, batch_y_val = next(iter(test_loader))
    batch_x_val = Variable(batch_x_val, requires_grad=False).cuda()
    batch_y_val = Variable(batch_y_val, requires_grad=False).cuda(non_blocking=True)
    batch_x_val, batch_y_val = batch_x_val.to(torch.float32), batch_y_val.to(torch.float32)

    batch_westimator_x_val, batch_westimator_y_val = pwn_model.westimator.prepare_input(batch_x_val, batch_y_val)


    batch_westimator_x_val = Variable(batch_westimator_x_val, requires_grad=False).cuda()
    batch_westimator_y_val = Variable(batch_westimator_y_val, requires_grad=False).cuda(non_blocking=True)

    architect.step(batch_x, batch_y, batch_x_val, batch_y_val,
                   batch_westimator_x,batch_westimator_y,batch_westimator_x_val,batch_westimator_y_val, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    prediction, prediction_ll = model(batch_x, batch_y, batch_westimator_x, batch_westimator_y)

    if use_smape:
      smape_adjust = 2  # Move all values into the positive space
      p_base_loss = lambda out, label: 2 * (torch.abs(out - label) /
                                            (torch.abs(out + smape_adjust) +
                                             torch.abs(label + smape_adjust))).mean(axis=1)

    else:
      p_base_loss = lambda out, label: nn.MSELoss(reduction='none')(out, label).mean(axis=1)

    prediction_loss = lambda error: error.mean()

    def ll_loss_pred(out, error):
      return (-1 * torch.logsumexp(out, dim=1) * (error ** -2)).mean() * 1e-4

    ll_loss = lambda out: -1 * torch.logsumexp(out, dim=1).mean()

    error = p_base_loss(prediction, batch_y)
    p_loss = prediction_loss(error)

    if model.train_rnn_w_ll:
      l_loss = ll_loss(prediction_ll)
      model.ll_weight_history.append(model.current_ll_weight)

      if model.weight_mse_by_ll is None:
        srnn_loss = (1 - model.current_ll_weight) * p_loss + model.current_ll_weight * l_loss
      else:
        local_ll = torch.logsumexp(prediction_ll, dim=1)
        local_ll = local_ll - local_ll.max()  # From 0 to -inf
        local_ll = local_ll / local_ll.min()  # From 0 to 1 -> low LL is 1, high LL is 0: Inverse Het
        local_ll = local_ll / local_ll.mean()  # Scale it to mean = 1

        if model.weight_mse_by_ll == 'het':
          # Het: low LL is 0, high LL is 1
          local_ll = local_ll.max() - local_ll

        srnn_loss = p_loss * (model.ll_weight - model.current_ll_weight) + \
                    model.current_ll_weight * (error * local_ll).mean()
    else:
      srnn_loss = p_loss
      l_loss = 0

    westimator_loss = ll_loss_pred(prediction_ll, error.detach())

    srnn_loss.backward()
    westimator_loss.backward()

    nn.utils.clip_grad_norm(model.parameters(), args.grad_clip)
    optimizer.step()

    if step % args.report_freq == 0:
      logging.info('train %03d %f', step, srnn_loss)

  return top1.avg, objs.avg


if __name__ == '__main__':
  torch.autograd.set_detect_anomaly(True)
  for r in range(10):
     main(r)
