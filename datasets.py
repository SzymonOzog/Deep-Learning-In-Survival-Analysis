from torch.utils.data import Dataset

class SurvDataset(Dataset):
    def __init__(self, x, events, time):
        self.x = x
        self.events = events
        self.time = time

    def __getitem__(self, index):
        return self.x[index], self.events[index], self.time[index]

    def __len__(self):
        return len(self.x)
    

class HitDataset(Dataset):
    def __init__(self, x, events, discrete_time, continous_time, mask):
        self.x = x
        self.events = events
        self.discrete_time = discrete_time
        self.continous_time = continous_time
        self.mask = mask

    def __getitem__(self, index):
        return self.x[index], self.events[index], self.discrete_time[index], self.continous_time[index], self.mask[index]

    def __len__(self):
        return len(self.x)