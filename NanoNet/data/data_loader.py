import numpy as np

class DataLoader():

    def __init__(self, dataset, shuffle, batch_size, drop_last) -> None:

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.sampler = Sampler(self.dataset, self.shuffle)
    
    def __iter__(self):
        batch_x = []
        batch_y = []
        
        for idx in self.sampler:
            x, y = self.dataset[idx]
            batch_x.append(x)
            batch_y.append(y)
            
            if len(batch_x) == self.batch_size:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []
        
        # Handle the last "leftover" batch
        if len(batch_x) > 0 and not self.drop_last:
            yield np.array(batch_x), np.array(batch_y)
    
    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


class Sampler():

    def __init__(self, data_source, shuffle) -> None:
        self.data_source = data_source
        self.shuffle = shuffle
    
    def __iter__(self):
        if self.shuffle:
            return iter(np.random.permutation(len(self.data_source)).tolist())
        else:
            return iter(range(len(self.data_source)))
        
    def __len__(self):
        return len(self.data_source)


