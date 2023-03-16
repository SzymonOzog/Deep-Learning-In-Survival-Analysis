import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import losses
import utils
import lifelines
from pycox.evaluation import EvalSurv

class SurvModelBase(nn.Module):
    def __init__(self, data, events_col, time_col):
        super(SurvModelBase, self).__init__()
        self.prepare_data(data, events_col, time_col)

    def prepare_data(self, data, events_col, time_col):
        data = data.sort_values(by=time_col, ascending=False)
        self.x = data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values

    def fit(self, epochs, train_index, valid_index, lr=0.001, verbose=True):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        history = {
            'loss': [],
            'val_loss': [],
            'c_index': [],
            'val_c_index': []
        }
        for epoch in range(epochs):
            self.train()
            closure = self.create_closure(train_index, optimizer)
            optimizer.step(closure)
            loss = closure()
            output = self(torch.tensor(self.x[train_index], dtype=torch.float))
            c_index = self.concordance_index(output.detach(), train_index)

            valid_loss, valid_c = self.validate(valid_index)
            if(verbose):
                print(f"Epoch {epoch} loss: {loss.item()}, concordance index: {c_index}\
                      valid loss: {valid_loss.item()}, valid concordance index: {valid_c}")
                
            history['loss'].append(loss.item())
            history['val_loss'].append(valid_loss.item())
            history['c_index'].append(c_index)
            history['val_c_index'].append(valid_c)
        return history

class SurvModel(SurvModelBase):
    def __init__(self, data, events_col, time_col):
        super(SurvModel, self).__init__(data, events_col, time_col)

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

    def create_closure(self, train_index, optimizer):
        def closure():
            optimizer.zero_grad()
            output = self(torch.tensor(self.x[train_index], dtype=torch.float))
            loss = losses.negative_likelihood_loss(output, torch.tensor(self.events[train_index], dtype=torch.float))
            loss.backward()
            return loss
        return closure
    
    def validate(self, valid_index):
        self.eval()
        output = self(torch.tensor(self.x[valid_index], dtype=torch.float))

        loss = losses.negative_likelihood_loss(output, torch.tensor(self.events[valid_index], dtype=torch.float))
        c_idx = self.concordance_index(output, valid_index)
        return loss, c_idx
    
    def concordance_index(self, output, index):
        return lifelines.utils.concordance_index(self.time[index], -output.detach(), event_observed=self.events[index])

class DeepHitModel(SurvModelBase):
    
    def __init__(self, data, events_col, time_col, time_bins):
        self.time_bins = time_bins
        super(DeepHitModel, self).__init__(data, events_col, time_col)

        self.dropout = nn.Dropout(0.1)

        self.fc1 = nn.Linear(len(self.x[0]), 90)
        self.bn1 = nn.BatchNorm1d(90)

        self.fc2 = nn.Linear(90, 64)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, time_bins)
    
    def prepare_data(self, data, events_col, time_col):
        self.data = data.sort_values(by=time_col, ascending=False)
        self.continous_time = self.data[time_col].values
        self.data[time_col] = utils.discretize_time(self.data[time_col], self.time_bins)

        self.mask = utils.create_mask(self.data[time_col].values, self.data[events_col].values)

        self.x = self.data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values
        
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
    
    def create_closure(self, train_index, optimizer):
            def closure():
                optimizer.zero_grad()
                output = self(torch.tensor(self.x[train_index], dtype=torch.float))
                loss = losses.deep_hit_loss(output, self.mask[train_index])
                loss.backward()
                return loss
            return closure
        
    def validate(self, valid_index):
        data = self.x[valid_index]
        mask = self.mask[valid_index]

        self.eval()
        output = self(torch.tensor(data, dtype=torch.float))

        loss = losses.deep_hit_loss(output, mask)
        c_idx = self.concordance_index(output, valid_index)
        return loss, c_idx
    
    def concordance_index(self, outputs, indices):
        surv = utils.create_surv_df(outputs.detach(), self.continous_time.max()/outputs.shape[1])
        ev = EvalSurv(surv, self.continous_time[indices], self.events[indices], censor_surv='km')
        return ev.concordance_td('antolini')