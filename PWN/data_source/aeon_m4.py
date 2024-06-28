from aeon.datasets import load_forecasting
import torch
from torch.utils.data import TensorDataset, random_split
import numpy as np

class M4Dataset:
    
    def __init__(self, train_len=24, test_len=6, subset='yearly') -> None:
        self.data = None
        self.norm_stats = {}
        self._train_len = train_len
        self._test_len = test_len
        self.df = load_forecasting(f'm4_{subset}_dataset', return_metadata=False)

    def prepare(self):
        x_data, y_data = [], []

        for _, row in self.df.iterrows():
            name = row['series_name']
            vals = row['series_value'].to_numpy()

            # padding
            if len(vals) < (self._train_len + self._test_len):
                pad_len = (self._train_len + self._test_len) - len(vals)
                vals = np.pad(vals, (pad_len, 0))

            x, y = vals[:self._train_len], vals[(self._train_len):(self._train_len + self._test_len)]

            mu, std = x.mean(), x.std()
            if std == 0:
                std = 1
            
            self.norm_stats[name] = (mu, std)

            x = (x - mu) / std
            y = (y - mu) / std

            x_data.append(x)
            y_data.append(y)

        
        self.data = TensorDataset(torch.from_numpy(np.array(x_data)), torch.from_numpy(np.array(y_data)))
        train_size = int(0.7 * len(self.data))
        test_size = len(self.data) - train_size
        self.train_data, self.test_data = random_split(self.data, [train_size, test_size])