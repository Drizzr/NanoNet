import os
import pickle
import gzip
from .dataset import Dataset
from NanoNet.utils import vectorized_result

class MNIST_DataSet_PKL(Dataset):
    def __init__(self, type='train'):
        super().__init__()
        self.type = type
        
        # FIND THE FILE DYNAMICALLY:
        # This looks for mnist.pkl.gz in the same folder as THIS file (examples.py)
        current_dir = os.path.dirname(__file__)
        self.data_path = os.path.join(current_dir, 'mnist.pkl.gz')
        
        self.data = self.load_data()

    def load_data(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Could not find MNIST data at {self.data_path}")
            
        with gzip.open(self.data_path, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            training_data, validation_data, test_data = u.load() 
        
        if self.type == 'train': return training_data
        elif self.type == 'validation': return validation_data
        elif self.type == 'test': return test_data

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        return self.data[0][idx], vectorized_result(self.data[1][idx], 10)