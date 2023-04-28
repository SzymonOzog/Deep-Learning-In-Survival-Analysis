import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import losses
import utils
import lifelines
from pycox.evaluation import EvalSurv

class SurvModelBase(nn.Module):
    def __init__(self, data, events_col, time_col, layers=[90, 64, 32], dropout=0.2, residual=False):
        super(SurvModelBase, self).__init__()
        self.residual = residual
        self.prepare_data(data, events_col, time_col)
        self.early_stopping = utils.EarlyStopping(patience=20, delta=0.001)
        self.layers = nn.ModuleList()


        for i in range(len(layers)):
            if i == 0:
                in_features = len(self.x[0])
            else:
                in_features = layers[i - 1]
            out_features = layers[i]
            if residual and i != 0 and i%2 == 0:
                in_features += layers[i - 2]
            self.layers.append(self.create_block(in_features, out_features, dropout))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if self.residual and i != 0 and i%2 == 0:
                x = torch.cat([x, residual], dim=1)
            x = layer(x)
            if self.residual and i%2 == 0:
                residual = x
        return x

    def create_block(self, in_features, out_features, dropout=0.2):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(out_features)
        )

    def prepare_data(self, data, events_col, time_col):
        data = data.sort_values(by=time_col, ascending=False)
        self.x = data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values

    def predict(self, indices):
        self.eval()
        output = self(torch.tensor(self.x[indices], dtype=torch.float))
        return output.detach()

        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, steps_per_epoch=1, epochs=epochs)
    def fit(self, epochs, train_index, valid_index, lr=0.001, verbose=True, weight_decay=0.01):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
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
            scheduler.step()
            loss = closure()
            output = self(torch.tensor(self.x[train_index], dtype=torch.float))
            c_index = self.concordance_index(output.detach(), train_index)

            valid_loss, valid_c = self.validate(valid_index)

            if self.early_stopping(-valid_loss, self):
                #load the last checkpoint with the best model
                self.load_state_dict(torch.load('checkpoint.pt'))
                break

            if(verbose):
                print(f"Epoch {epoch} loss: {loss.item()}, concordance index: {c_index}\
                      valid loss: {valid_loss.item()}, valid concordance index: {valid_c}\
                      learning rate: {scheduler.get_last_lr()}")
                
            history['loss'].append(loss.item())
            history['val_loss'].append(valid_loss.item())
            history['c_index'].append(c_index)
            history['val_c_index'].append(valid_c)
        return history

class SurvModel(SurvModelBase):
    def __init__(self, data, events_col, time_col, layers=[90, 64, 32], dropout=0.2, residual=False):
        super(SurvModel, self).__init__(data, events_col, time_col, layers, dropout, residual)
        if residual and len(layers) % 2 == 0:
            self.layers.append(nn.Linear(layers[-1] + layers[-3], 1))
        else:
            self.layers.append(nn.Linear(layers[-1], 1))


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
    
    def __init__(self, data, events_col, time_col, time_bins, interpolation_steps = 10, layers=[90, 64, 32], dropout=0.2, residual=False):
        self.time_bins = time_bins
        self.interpolation_steps = interpolation_steps

        super(DeepHitModel, self).__init__(data, events_col, time_col, layers, dropout, residual)
        
        if residual and len(layers) % 2 == 0:
            self.layers.append(nn.Linear(layers[-1] + layers[-3], time_bins))
        else:
            self.layers.append(nn.Linear(layers[-1], time_bins))
        self.layers.append(nn.Softmax(dim=1))
    
    def prepare_data(self, data, events_col, time_col):
        self.data = data
        self.continous_time = self.data[time_col].values
        self.data[time_col] = utils.discretize_time(self.data[time_col], self.time_bins)

        self.mask = utils.create_mask(self.data[time_col].values, self.data[events_col].values)

        self.x = self.data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values
    
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
        surv = utils.create_surv_df(outputs.detach(), self.continous_time.max()/outputs.shape[1], self.interpolation_steps)
        ev = EvalSurv(surv, self.continous_time[indices], self.events[indices], censor_surv='km')
        return ev.concordance_td('antolini')