'''
Opens text files and creates dataloader objects for training a model.
'''
import torch
from torch.utils.data import Dataset, DataLoader
import utils

class Dataset(Dataset):
    def __init__(self, data):
        self.data = torch.tensor(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #torch tensor list of lists

        return self.data[index]

'''
    Creates dataloaders for the training, validation, and test datasets.
    
    Parameters: none
    
    Returns:
        training_dataloader, validation_dataloader, test_dataloader
'''
def create_dataloaders(train_data, validation_data, test_data, batch_size):

    #Initialize datasets, replacing words with indexes or '0,' as appropriate
    training_dataset = Dataset(train_data)
    validation_dataset = Dataset(validation_data)
    test_dataset = Dataset(test_data)

    #Create dataloaders
    training_dataloader = DataLoader(training_dataset, batch_size=batch_size,shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size,shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True)
    return training_dataloader, validation_dataloader, test_dataloader
