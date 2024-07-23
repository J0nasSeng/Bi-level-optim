from aeon.datasets import load_forecasting
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from losses import SMAPE
#from neuralforecast.losses.pytorch import SMAPE
from spectral_gru import SpectralGRUNet

class M4Dataset:

    def __init__(self, train_len=24, test_len=6) -> None:
        self.data = None
        self.norm_stats = {}
        self._train_len = train_len
        self._test_len = test_len
        self.df = load_forecasting('m4_quarterly_dataset', return_metadata=False)

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
hidden_size = 196
output_size = 6
num_layers = 3
device = torch.device('cuda:7')

model = SpectralGRUNet(hidden_size, output_size, device, num_layers).to(device)
#criterion = nn.MSELoss()
criterion = SMAPE()
optimizer = optim.Adam(model.parameters(), lr=0.001)
#lr_scheduler = optim.lr_scheduler.LinearLR()

for name, param in model.named_parameters():
    if 'weight' in name:
        nn.init.xavier_uniform_(param.data)


# 3. Train the model
num_epochs = 1000

for epoch in range(num_epochs):

    avg_loss = 0

    for i, (x_batch, y_batch) in enumerate(dataloader):
        x_batch = x_batch.unsqueeze(-1)  # Adding feature dimension
        y_batch = y_batch.unsqueeze(-1)  # Adding feature dimension
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
