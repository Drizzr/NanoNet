from .dataset import Dataset
import pickle
import gzip
from NanoNet.utils import vectorized_result

class MNIST_DataSet_PKL(Dataset):

    def __init__(self, data_path : str, type : str ='train'):
        super().__init__()

        self.data_path = data_path
        self.type = type

        self.data = self.load_data()


    def load_data(self):
        with gzip.open('mnist_example/data/mnist.pkl.gz', 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            training_data, validation_data, test_data = u.load() 
        
        if self.type == 'train':
            return training_data
        elif self.type == 'validation':
            return validation_data
        elif self.type == 'test':
            return test_data

    def __len__(self):
        return len(self.data[0])
    
    def __getitem__(self, idx):
        return self.data[0][idx], vectorized_result(self.data[1][idx], 10) # format: (785), (10)

