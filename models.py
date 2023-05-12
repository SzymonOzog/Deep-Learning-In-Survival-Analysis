import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import losses
import utils
import math
import lifelines
from pycox.evaluation import EvalSurv
from datasets import *

class SurvModelBase(nn.Module):
    def __init__(self, data, events_col, time_col, batch_size=64, layers=[90, 64, 32], dropout=0.2, residual=False):
        super(SurvModelBase, self).__init__()

        residual = residual and len(layers) > 2

        self.batch_size = batch_size
        self.residual = residual
        self.prepare_data(data, events_col, time_col)
        self.early_stopping = utils.EarlyStopping(patience=100, delta=0.001)
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
        for col in data.columns:
            data[col] = data[col].astype(float)
        data = data.sort_values(by=time_col, ascending=False)
        self.x = data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values

    def predict(self, indices):
        self.eval()
        output = self(torch.tensor(self.x[indices], dtype=torch.float))
        return output.detach()

    def fit(self, epochs, train_index, valid_index, lr=0.001, verbose=True, weight_decay=0.01):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        history = {
            'loss': [],
            'val_loss': [],
            'c_index': [],
            'val_c_index': []
        }
        train_dataloader, valid_dataloader = self.create_data_loaders(train_index, valid_index)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, 
                                                        steps_per_epoch=math.ceil(len(train_index)/self.batch_size), epochs=epochs)
        for epoch in range(epochs):
            for batch in train_dataloader:
                # convert batch to float
                batch = [item.float() for item in batch]
                self.train()
                optimizer.zero_grad()
                loss = self.get_loss(batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

            loss, c_index = self.validate(train_dataloader)
            valid_loss, valid_c = self.validate(valid_dataloader)

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
    def __init__(self, data, events_col, time_col, batch_size=64, layers=[90, 64, 32], dropout=0.2, residual=False):
        super(SurvModel, self).__init__(data, events_col, time_col, batch_size, layers, dropout, residual)
        if self.residual and len(layers) % 2 == 0:
            self.layers.append(nn.Linear(layers[-1] + layers[-3], 1))
        else:
            self.layers.append(nn.Linear(layers[-1], 1))

    def create_data_loaders(self, train_index, valid_index):
        train_dataset = SurvDataset(self.x[train_index], self.events[train_index], self.time[train_index])
        valid_dataset = SurvDataset(self.x[valid_index], self.events[valid_index], self.time[valid_index])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        return train_dataloader, valid_dataloader


    def get_loss(self, batch):
        # sort batch by time descending
        index = torch.argsort(batch[2], descending=True)
        batch = [item[index] for item in batch]
        output = self(batch[0])
        return losses.negative_likelihood_loss(output, batch[1])
    
    def validate(self, dataloader):
        full_output = []
        full_batch = [[], [], []]
        self.eval()
        for batch in dataloader:
            # convert batch to float
            batch = [item.float() for item in batch]
            output = self(batch[0])
            full_output.append(output.detach())
            for i in range(len(batch)):
                full_batch[i].append(batch[i])            

        full_output = torch.cat(full_output)
        for i in range(len(full_batch)):
            full_batch[i] = torch.cat(full_batch[i])
        
        # sort output and batch by time descending
        indices = torch.argsort(full_batch[2], descending=True)
        full_output = full_output[indices]
        for i in range(len(full_batch)):
            full_batch[i] = full_batch[i][indices]

        loss = losses.negative_likelihood_loss(full_output, full_batch[1])
        c_idx = self.concordance_index(full_output, full_batch)
        return loss, c_idx
    
    
    def concordance_index(self, output, batch):
        return lifelines.utils.concordance_index(batch[2], -output.detach(), event_observed=batch[1])

class DeepHitModel(SurvModelBase):
    
    def __init__(self, data, events_col, time_col, time_bins, batch_size=64, interpolation_steps = 10, layers=[90, 64, 32], dropout=0.2, residual=False):
        self.time_bins = time_bins
        self.interpolation_steps = interpolation_steps

        super(DeepHitModel, self).__init__(data, events_col, time_col, batch_size, layers, dropout, residual)
        
        if self.residual and len(layers) % 2 == 0:
            final_layer = nn.Linear(layers[-1] + layers[-3], time_bins)
        else:
            final_layer = nn.Linear(layers[-1], time_bins)

        self.layers.append(nn.Sequential(
            final_layer,
            nn.Softmax(dim=1))
            )
    
    def prepare_data(self, data, events_col, time_col):
        self.data = data
        #convert data to float
        for col in self.data.columns:
            self.data[col] = self.data[col].astype(float)
        self.continous_time = self.data[time_col].values
        self.data[time_col] = utils.discretize_time(self.data[time_col], self.time_bins)

        self.mask = utils.create_mask(self.data[time_col].values, self.data[events_col].values)

        self.x = self.data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values

    def create_data_loaders(self, train_index, valid_index):
        train_dataset = HitDataset(self.x[train_index], self.events[train_index], self.time[train_index], self.continous_time[train_index], self.mask[train_index])
        valid_dataset = HitDataset(self.x[valid_index], self.events[valid_index], self.time[valid_index], self.continous_time[valid_index], self.mask[valid_index])
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        return train_dataloader, valid_dataloader
    
    def get_loss(self, batch):
            output = self(batch[0])
            return losses.deep_hit_loss(output, batch[4])
        
    def validate(self, dataloader):
        full_output = []
        full_batch = [[], [], [], [], []]
        self.eval()
        for batch in dataloader:
            batch = [item.float() for item in batch]
            output = self(batch[0])
            full_output.append(output.detach())
            for i in range(len(batch)):
                full_batch[i].append(batch[i])    

        full_output = torch.cat(full_output) 
        for i in range(len(full_batch)):
            full_batch[i] = torch.cat(full_batch[i])       

        loss = losses.deep_hit_loss(full_output, full_batch[4])
        c_idx = self.concordance_index(full_output, full_batch)
        return loss, c_idx
    
    def concordance_index(self, outputs, batch):
        surv = utils.create_surv_df(outputs.detach(), self.continous_time.max()/outputs.shape[1], self.interpolation_steps)
        ev = EvalSurv(surv, batch[3].detach().numpy(), batch[1].detach().numpy(), censor_surv='km')
        return ev.concordance_td('antolini')