import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import losses
import utils
from pycox.evaluation import EvalSurv

class SurvModel(nn.Module):
    def __init__(self, data, events_col, time_col):
        super(SurvModel, self).__init__()

        data = data.sort_values(by=time_col, ascending=False)
        self.x = data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values

        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(len(self.x[0]), 90)
        self.bn1 = nn.BatchNorm1d(90)

        self.fc2 = nn.Linear(90, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.fc4(x)
        return x

    def fit(self, epochs, lr=0.001, verbose=True):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        def closure():
            optimizer.zero_grad()
            output = self(torch.tensor(self.x, dtype=torch.float))
            loss = losses.negative_likelihood_loss(output, torch.tensor(self.events, dtype=torch.float))
            loss.backward()
            return loss
        for epoch in range(epochs):
            optimizer.step(closure)
            if(verbose):
                loss = closure()
                print(f"Epoch {epoch} loss: {loss.item()}/{losses.negative_likelihood_loss(self(torch.tensor(self.x, dtype=torch.float)), torch.tensor(self.events, dtype=torch.float))}")

class DeepHitModel(nn.Module):
    
    def __init__(self, data, events_col, time_col, time_bins):
        super(DeepHitModel, self).__init__()

        self.data = data.sort_values(by=time_col, ascending=False)

        self.continous_time = self.data[time_col].values
        self.data[time_col] = utils.discretize_time(self.data[time_col], time_bins)

        self.mask = utils.create_mask(self.data[time_col].values, self.data[events_col].values)

        self.x = self.data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values

        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(len(self.x[0]), 90)
        self.bn1 = nn.BatchNorm1d(90)

        self.fc2 = nn.Linear(90, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, time_bins)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.bn3(x)
        x = self.fc4(x)
        return F.softmax(x, dim=1)
    
    def fit(self, epochs, train_index, valid_index, lr=0.001, verbose=True):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        history = {
            'loss': [],
            'val_loss': [],
            'c_index': [],
            'val_c_index': []
        }
        def closure():
            optimizer.zero_grad()
            output = self(torch.tensor(self.x[train_index], dtype=torch.float))
            loss = losses.deep_hit_loss(output, self.mask[train_index])
            loss.backward()
            return loss
        for epoch in range(epochs):
            self.train()
            optimizer.step(closure)
            loss = closure()
            output = self(torch.tensor(self.x[train_index], dtype=torch.float))
            surv = utils.create_surv_df(output.detach(), self.continous_time.max()/output.shape[1])
            ev = EvalSurv(surv, self.continous_time[train_index], self.events[train_index], censor_surv='km')
            valid_loss, valid_c = self.validate(self.x[valid_index], self.events[valid_index], self.continous_time[valid_index], self.mask[valid_index])
            if(verbose):
                print(f"Epoch {epoch} loss: {loss.item()}, concordance index: {ev.concordance_td('antolini')}\
                      valid loss: {valid_loss.item()}, valid concordance index: {valid_c}")
            history['loss'].append(loss.item())
            history['val_loss'].append(valid_loss.item())
            history['c_index'].append(ev.concordance_td('antolini'))
            history['val_c_index'].append(valid_c)
        return history
        
    def validate(self, data, events, time, mask):
        self.eval()
        output = self(torch.tensor(data, dtype=torch.float))
        surv = utils.create_surv_df(output.detach(), self.continous_time.max()/output.shape[1])
        ev = EvalSurv(surv, time, events, censor_surv='km')
        loss = losses.deep_hit_loss(output, mask)
        c_idx = ev.concordance_td('antolini')
        return loss, c_idx