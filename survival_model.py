import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

class IntervalCensoredDataset(Dataset):
    def __init__(self, features, lower_times, upper_times):
        self.features = torch.FloatTensor(features)
        self.lower_times = torch.FloatTensor(lower_times)
        self.upper_times = torch.FloatTensor(upper_times)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'lower_time': self.lower_times[idx],
            'upper_time': self.upper_times[idx]
        }

class DeepSurvivalNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32]):
        super(DeepSurvivalNet, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.hazard_layer = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        features = self.network(x)
        hazard = torch.exp(self.hazard_layer(features))
        return hazard

def interval_censored_loss(hazard_pred, lower_time, upper_time):
    surv_lower = torch.exp(-hazard_pred * lower_time)
    surv_upper = torch.exp(-hazard_pred * upper_time)
    interval_prob = surv_lower - surv_upper
    eps = 1e-7
    interval_prob = torch.clamp(interval_prob, min=eps)
    loss = -torch.log(interval_prob).mean()
    return loss

def train_model(model, train_loader, val_loader, epochs=100, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            features = batch['features']
            lower_time = batch['lower_time']
            upper_time = batch['upper_time']
            
            optimizer.zero_grad()
            hazard_pred = model(features)
            loss = interval_censored_loss(hazard_pred, lower_time, upper_time)
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                features = batch['features']
                lower_time = batch['lower_time']
                upper_time = batch['upper_time']
                
                hazard_pred = model(features)
                loss = interval_censored_loss(hazard_pred, lower_time, upper_time)
                val_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss/len(train_loader):.4f}')
            print(f'Val Loss: {val_loss/len(val_loader):.4f}')