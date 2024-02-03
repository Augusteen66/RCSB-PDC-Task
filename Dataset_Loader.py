import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, TensorDataset, Dataset

train_x = pd.read_csv("data/pairs_train.csv")
train_y = pd.read_csv("data/labels_train.csv")
test_x = pd.read_csv("data/pairs_test.csv")
test_y = pd.read_csv("data/labels_test.csv")

class customDataset(Dataset):
    
    def __init__(self, data, labels, transform=False):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        
        x = torch.tensor(self.data.loc[index, ['Geo_Distance', 'Zernike_Distance']].values, dtype=torch.float) 
        y = torch.tensor(self.labels.iloc[index, 0], dtype=torch.float)
        
        sample = {'x': x, 'y': y}
        
        
        return sample
    
def getData(batch_size, use_weighted_sampler=True):
    train_data_object = customDataset(train_x, train_y)
    test_data_object = customDataset(test_x, test_y)

    # Calculate class weights for the training dataset
    class_counts = torch.bincount(torch.tensor(train_y.squeeze()))
    class_weights = 1.0 / class_counts.float()

    # Create a sampler using the class weights if specified
    if use_weighted_sampler:
        weights = class_weights[train_y.squeeze()]
        sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
        train_data_loader = DataLoader(train_data_object, batch_size=batch_size, sampler=sampler)
    else:
        train_data_loader = DataLoader(train_data_object, batch_size=batch_size, shuffle=True)

    test_data_loader = DataLoader(test_data_object, batch_size=batch_size, shuffle=True)

    return train_data_loader, test_data_loader