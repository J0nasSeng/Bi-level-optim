from data_source import BasicSelect, Mackey, ReadPowerPKL, ReadExchangePKL, ReadM4, ReadWikipediaPKL, ReadSolarPKL
from preprocessing import *
from model import PWNv2
from model.cwspn import CWSPN, CWSPNConfig
from model.wein import WEin, WEinConfig
from model.wein.EinsumNetwork.ExponentialFamilyArray import NormalArray, MultivariateNormalArray, BinomialArray
from evaluation import *
from util.plot import plot_experiment, cmp_plot, ll_plot
from util.dataset import split_dataset
from util.store_experiments import save_experiment
from model.transformer import TransformerConfig
import pickle
from datetime import datetime
import torch
from data_source import M4Dataset
from util.losses import SMAPE
from torch.utils.data import DataLoader


plot_base_path = 'res/plots/'
model_base_path = 'res/models/'
experiments_base_path = 'res/experiments/'

#config = SpectralRNNConfig()
#config.use_searched_srnn = True #Choose if you want to use architecture searched srnn
## config.normalize_fft = True
#config.use_add_linear = False
#config.rnn_layer_config.use_gated = True
#config.rnn_layer_config.use_cg_cell = False
#config.rnn_layer_config.use_residual = True
#config.rnn_layer_config.learn_hidden_init = False
#config.rnn_layer_config.use_linear_projection = True
#config.rnn_layer_config.dropout = 0.1
#
#config.hidden_dim = 128
#config.rnn_layer_config.n_layers = 2      # Set to 12 if you want to run SRNN+
#
#config.use_cached_predictions = False

config_w = WEinConfig()
config_w.exponential_family = NormalArray #NormalArray #MultivariateNormalArray
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
config_c.use_searched_cwspn = False #Choose if you want to use architecture searched cwspn

config_t = TransformerConfig()

manual_split = True

m4_key = 'Yearly'
m4_settings = {
    'Hourly': {'window_size': 24, 'fft_compression': 2, 'context_timespan': int(20 * 24),
               'prediction_timespan': int(2 * 24), 'timespan_step': int(.5 * 24)},  # 700 Min Context
    'Daily': {'window_size': 14, 'fft_compression': 2, 'context_timespan': int(5 * 14),
              'prediction_timespan': int(1 * 14), 'timespan_step': int(.5 * 14)},  # 93 Min Context
    'Weekly': {'window_size': 14, 'fft_compression': 2, 'context_timespan': int(4.5 * 14),
               'prediction_timespan': int(14), 'timespan_step': int(.5 * 14)},  # 80 Min Context
    'Monthly': {'window_size': 18, 'fft_compression': 1, 'context_timespan': int(6 * 18),
                'prediction_timespan': int(1 * 18), 'timespan_step': int(.5 * 18)},  # 42 Min Context
    'Quarterly': {'window_size': 8, 'fft_compression': 1, 'context_timespan': int(4 * 8),
                  'prediction_timespan': int(1 * 8), 'timespan_step': int(.5 * 8)},  # 16 Min Context
    'Yearly': {'window_size': 6, 'fft_compression': 1, 'context_timespan': int(4 * 6),
               'prediction_timespan': int(1 * 6), 'timespan_step': int(.5 * 6)}  # 13 Min Context
}

#config.window_size = m4_settings[m4_key]['window_size']#96#m4_settings[m4_key]['window_size']#96
#config.fft_compression = m4_settings[m4_key]['fft_compression']#4#m4_settings[m4_key]['fft_compression']#4

hidden_size = 128
output_size = 6

exchange_context_timespan = 6*30 #6*30
exchange_prediction_timespan = 30
exchange_timespan_step = 15 #20
#config.window_size = 60 #20
#config.fft_compression = 2 #2

wiki_context_timespan = 6*30
wiki_prediction_timespan = 30
wiki_timespan_step = 30 #20
#config.window_size = 60 #20
#config.fft_compression = 2 #2

solar_context_timespan = 30*24
solar_prediction_timespan = 24
solar_timespan_step = 10*24
#config.window_size = 24
#config.fft_compression = 2

#config.window_size = m4_settings[m4_key]['window_size']
#config.fft_compression = m4_settings[m4_key]['fft_compression']

#Define experiment to run, ReadM4 requieres the m4key as an additional argument
device = torch.device('cuda:0')
#experiments = [
#  (ReadM4(m4_key), ZScoreNormalization((0,), 3, 2, [True, True, True, False], min_group_size=0,
#                                      context_timespan=int(m4_settings[m4_key]['context_timespan']), prediction_timespan=int(m4_settings[m4_key]['prediction_timespan']),
#                                      timespan_step=int(m4_settings[m4_key]['timespan_step']), single_group=False, multivariate=False, retail=False),
#  PWNv2(hidden_size, output_size, m4_settings[m4_key]['fft_compression'], m4_settings[m4_key]['window_size'], 0.5, device,
#        config_c, train_spn_on_gt=False, train_spn_on_prediction=True, train_rnn_w_ll=False,
#         always_detach=True,use_transformer=False,smape_target=True),
#  [SMAPE(),CorrelationError(), MAE(), MSE(), RMSE()],
#  {'train': False, 'reversed': False, 'll': False, 'single_ll': False, 'mpe': False},
#  None, False)
#]

dataset = M4Dataset()
dataset.prepare()

dataloader = DataLoader(dataset.train_data, batch_size=256, shuffle=True)
model = PWNv2(hidden_size, output_size, m4_settings[m4_key]['fft_compression'], m4_settings[m4_key]['window_size'],
              0.5, device, config_c, num_srnn_layers=2, train_spn_on_gt=False, train_spn_on_prediction=True,
              smape_target=True, use_transformer=True)

model.train(dataloader, epochs=100)

test_loader = DataLoader(dataset.test_data, batch_size=1024)

preds, lls, gt = model.predict_v2(dataloader)

score = SMAPE()

result = score(gt, preds)

print(f"SMAPE={result.item()}")