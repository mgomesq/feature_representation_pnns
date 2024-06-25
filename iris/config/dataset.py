'''Definition for the dataset and data transformations used.'''

import torch
from torch.utils.data import Dataset

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class IrisDataset():
    '''
    Definition of an Iris dataset. Loads the model from sklearn and applies
    some small data manipulation on it, such as scaling, changing feature
    order and sending to a device.

    Arguments
    ---------
    feature_order - ordering of features. Must be a list.
                    default order is [0,1,2,3]
    device - str, device to send data to
    '''
    def __init__(self, feature_order=None, device='cpu'):

        self.dataset = datasets.load_iris()

        x_iris = self.dataset['data']

        if feature_order:
            x_iris = x_iris[:, feature_order]

        y_iris = self.dataset['target']

        data_split = train_test_split(
                        x_iris,
                        y_iris,
                        test_size=0.30,
                        random_state=42
                        )

        x_train, x_test, y_train, y_test = data_split

        self.scaler = StandardScaler()

        x_train = self.scaler.fit_transform(x_train)
        x_test = self.scaler.transform(x_test)

        self.train = self.SimpleDataset(x_train, y_train, device=device)
        self.test = self.SimpleDataset(x_test, y_test, device=device)

    class SimpleDataset(Dataset):
        ''' Inner class defining a simple dataset behaviour. It is used to
        feed data correctly in train/test dataset for IrisDataset.
        '''
        def __init__(self, x, y, device='cpu'):
            self.x = torch.tensor(x, device=device)
            self.y = torch.tensor(y, device=device, dtype=torch.long)

        def __len__(self):
            return len(self.y)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]
