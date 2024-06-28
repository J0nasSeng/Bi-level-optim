from preprocessing import *
from model import PWNv2
from model.cwspn import CWSPNConfig
from model.wein import WEinConfig
from model.wein.EinsumNetwork.ExponentialFamilyArray import NormalArray
from evaluation import *
from model.transformer import TransformerConfig
import torch
from data_source import M4Dataset
from util.losses import SMAPE
from torch.utils.data import DataLoader
import time
import json
import os

plot_base_path = 'res/plots/'
model_base_path = 'res/models/'
experiments_base_path = 'res/experiments/'

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

dataset_key = 'm4_Hourly'
hyperparams = {
    'm4_Hourly': {'window_size': 24, 'fft_compression': 2, 'context_timespan': int(20 * 24),
               'prediction_timespan': int(2 * 24), 'timespan_step': int(.5 * 24)},  # 700 Min Context
    'm4_Daily': {'window_size': 14, 'fft_compression': 2, 'context_timespan': int(5 * 14),
              'prediction_timespan': int(1 * 14), 'timespan_step': int(.5 * 14)},  # 93 Min Context
    'm4_Weekly': {'window_size': 14, 'fft_compression': 2, 'context_timespan': int(4.5 * 14),
               'prediction_timespan': int(14), 'timespan_step': int(.5 * 14)},  # 80 Min Context
    'm4_Monthly': {'window_size': 18, 'fft_compression': 1, 'context_timespan': int(6 * 18),
                'prediction_timespan': int(1 * 18), 'timespan_step': int(.5 * 18)},  # 42 Min Context
    'm4_Quarterly': {'window_size': 8, 'fft_compression': 1, 'context_timespan': int(4 * 8),
                  'prediction_timespan': int(1 * 8), 'timespan_step': int(.5 * 8)},  # 16 Min Context
    'm4_Yearly': {'window_size': 6, 'fft_compression': 1, 'context_timespan': int(4 * 6),
               'prediction_timespan': int(1 * 6), 'timespan_step': int(.5 * 6)},  # 13 Min Context
    "solar": {'window_size': 24, 'fft_compression': 2, 'context_timespan': 30*24,
               'prediction_timespan': 24, 'timespan_step': 10*24},
    "exchange": {'window_size': 60, 'fft_compression': 2, 'context_timespan': 6*30,
               'prediction_timespan': 30, 'timespan_step': 15},
    "wiki": {'window_size': 60, 'fft_compression': 2, 'context_timespan': 6*30,
               'prediction_timespan': 30, 'timespan_step': 30}
}

#config.window_size = hyperparams[dataset_key]['window_size']#96#hyperparams[dataset_key]['window_size']#96
#config.fft_compression = hyperparams[dataset_key]['fft_compression']#4#hyperparams[dataset_key]['fft_compression']#4
use_transformer = True
hidden_size = 128 if use_transformer else 64
output_size = 6
learning_rate = 0.0004 if use_transformer else 0.004
epochs = 10

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

#config.window_size = hyperparams[dataset_key]['window_size']
#config.fft_compression = hyperparams[dataset_key]['fft_compression']

#Define experiment to run, ReadM4 requieres the m4key as an additional argument
device = torch.device('cuda:3')

m4_key = dataset_key[3:].lower()
dataset = M4Dataset(train_len=hyperparams[dataset_key]['context_timespan'], test_len=hyperparams[dataset_key]['prediction_timespan'], subset=m4_key)
dataset.prepare()

seeds = list(range(10))
for s in seeds:
    torch.manual_seed(s)

    dataloader = DataLoader(dataset.train_data, batch_size=256, shuffle=True)
    model = PWNv2(hidden_size, hyperparams[dataset_key]['prediction_timespan'], hyperparams[dataset_key]['fft_compression'], 
                  hyperparams[dataset_key]['window_size'], 0.5, device, config_c, num_srnn_layers=2, train_spn_on_gt=False, train_spn_on_prediction=True,
                smape_target=True, use_transformer=use_transformer)

    start_time = time.time()
    model.train(dataloader, epochs=epochs, lr=learning_rate)
    end_time = time.time()

    test_loader = DataLoader(dataset.test_data, batch_size=1024)

    preds, lls, gt = model.predict_v2(dataloader)

    score = SMAPE()

    result = score(gt, preds)

    log_dict = {
        "neural_model": "transformer" if use_transformer else "srnn",
        "pc": "cwspn",
        "test_smape": result.item(),
        "hyperparameters": {
            "hidden_size": hidden_size,
            "output_size": output_size,
            "ds_hyperparams": hyperparams
        },
        "execution_time": (end_time - start_time) / 3600,
        "seed": s,
        "learning_rate": learning_rate
    }

    log_file = f"{log_dict['neural_model']}_{log_dict['pc']}_{dataset_key}_{s}.json"

    with open(os.path.join(experiments_base_path, log_file), 'w') as f:
        json.dump(log_dict, f, indent=4)
    