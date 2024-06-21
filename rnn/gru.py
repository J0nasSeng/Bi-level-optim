from aeon.datasets import load_forecasting
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import numpy as np
from neuralforecast.losses.pytorch import SMAPE

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

# 2. Define the GRU model
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

input_size = 1
hidden_size = 128
output_size = 6
num_layers = 2
device = torch.device('cuda:3')

model = GRUNet(input_size, hidden_size, output_size, num_layers).to(device)
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
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch.squeeze())

        avg_loss += loss.item() #/ len(x_batch)
        
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
        outputs = model(x_batch)
        loss = criterion(outputs, y_batch.squeeze())
        test_smape += loss.item()

test_smape = test_smape / len(test_loader)

print(f"SMAPE: {test_smape}")

# TODO: check SRNN repo