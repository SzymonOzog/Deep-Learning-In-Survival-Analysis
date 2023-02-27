import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lifelines.utils import concordance_index

class SurvModel(nn.Module):
    def __init__(self, data, events_col, time_col):
        super(SurvModel, self).__init__()

        data = data.sort_values(by=time_col)
        self.x = data.drop([events_col, time_col], axis=1).values
        self.events = data[events_col].values
        self.time = data[time_col].values

        self.fc1 = nn.Linear(len(self.x[0]), 128)
        self.fc2 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.fc4(x)
        return x

    def fit(self, epochs, lr=0.001, verbose=True):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = self(torch.tensor(self.x, dtype=torch.float))
            loss = self.negative_likelihood_loss(output, torch.tensor(self.events, dtype=torch.float))
            loss.backward()
            optimizer.step()
            if((epoch/epochs) % 0.1 == 0 and verbose):   
                print(f"Epoch {epoch} loss: {loss.item()}, concordance index: {concordance_index(self.time, torch.exp(output.detach().squeeze()), self.events)}")
 
    def negative_likelihood_loss(self, y_pred, events):
        y_pred = y_pred.squeeze()
        hazard_ratio = torch.exp(y_pred)
        risk = torch.cumsum(hazard_ratio, dim=0)
        log_risk = torch.log(risk)
        uncensored_likelihood = y_pred - log_risk
        censored_likelihood = uncensored_likelihood * events
        cum_loss = -torch.sum(censored_likelihood)
        total_events = torch.sum(events)
        return cum_loss / total_events

