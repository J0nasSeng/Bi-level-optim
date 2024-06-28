from aeon.datasets import load_forecasting
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from model.transformer.losses import SMAPE
#from neuralforecast.losses.pytorch import SMAPE
from model.transformer.transformer_predictor import TransformerNet, TransformerConfig

class M4Dataset:

    def __init__(self, train_len=24, test_len=6) -> None:
        self.data = None
        self.norm_stats = {}
        self._train_len = train_len
        self._test_len = test_len
        self.df = load_forecasting('m4_yearly_dataset', return_metadata=False)

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

   
dataset = M4Dataset()
dataset.prepare()
dataloader = DataLoader(dataset.train_data, batch_size=256, shuffle=True)


input_size = 1
hidden_size = 64
output_size = 6
window_size = 6
fft_compression = 1
overlap = 0.5
num_layers = 2
device = torch.device('cuda:3')

trans_cfg = TransformerConfig(normalize_fft=True, window_size=window_size, dropout=0.1,
                  fft_compression=fft_compression, hidden_dim=hidden_size, embedding_dim=32,
                  num_enc_dec=2, is_complex=False, native_complex=False)
trans_cfg.step_width = int(window_size * overlap)
trans_cfg.value_dim = window_size // 2 + 1

trans_cfg.compressed_value_dim = trans_cfg.value_dim // fft_compression
trans_cfg.removed_freqs = trans_cfg.value_dim - trans_cfg.compressed_value_dim
trans_cfg.input_dim = trans_cfg.compressed_value_dim
model = TransformerNet(trans_cfg, trans_cfg.input_dim * 2, trans_cfg.hidden_dim,
                                  trans_cfg.input_dim * 2, trans_cfg.q, trans_cfg.k, trans_cfg.heads,
                                  trans_cfg.num_enc_dec, attention_size=trans_cfg.attention_size,
                                  dropout=trans_cfg.dropout, chunk_mode=trans_cfg.chunk_mode, pe=trans_cfg.pe,
                                  complex=trans_cfg.is_complex, native_complex=trans_cfg.native_complex, device=device).to(device)
#criterion = nn.MSELoss()
criterion = SMAPE()
#optimizer = optim.Adam(model.parameters(), lr=0.001)
optimizer = optim.RMSprop(model.parameters(), 0.0004)
#lr_scheduler = optim.lr_scheduler.StepLR(optimizer, 5, 0.98)


# 3. Train the model
num_epochs = 4000

for epoch in range(num_epochs):

    avg_loss = 0

    for i, (x_batch, y_batch) in enumerate(dataloader):
        #x_batch = x_batch.unsqueeze(-1)  # Adding feature dimension
        #y_batch = y_batch.unsqueeze(-1)  # Adding feature dimension
        x_batch = x_batch.to(torch.float32)
        y_batch = y_batch.to(torch.float32)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        outputs, _ = model(x_batch, y_batch)
        loss = criterion(outputs, y_batch.squeeze())

        avg_loss += loss.item() / len(dataloader)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        #lr_scheduler.step()
        
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')


test_loader = DataLoader(dataset.test_data, batch_size=256, shuffle=True)

test_smape = 0.0

with torch.no_grad():
    for x, y in test_loader:
        outputs, _ = model(x_batch, y_batch)
        loss = criterion(outputs, y_batch.squeeze())
        test_smape += loss.item()

test_smape = test_smape / len(test_loader)

print(f"SMAPE: {test_smape}")
