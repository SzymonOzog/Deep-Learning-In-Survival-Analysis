import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import losses
import utils

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


