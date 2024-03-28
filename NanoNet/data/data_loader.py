import numpy as np

class DataLoader():

    def __init__(self, dataset, shuffle, batch_size, drop_last) -> None:

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last

        self.sampler = Sampler(self.dataset, self.shuffle)
    
    def __iter__(self):
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    x_batch = [self.dataset[next(sampler_iter)][0] for _ in range(self.batch_size)]
                    y_batch = [self.dataset[next(sampler_iter)][1] for _ in range(self.batch_size)]
                    yield np.array(x_batch), np.array(y_batch)
                except StopIteration:
                    break
        else:
            batch_x = [0] * self.batch_size
            batch_y = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch_x[idx_in_batch] = self.dataset[idx][0]
                batch_y[idx_in_batch] = self.dataset[idx][1]
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    idx_in_batch = 0
                    batch_x = [0] * self.batch_size
                    batch_y = [0] * self.batch_size
            if idx_in_batch > 0:
                yield np.array(batch_x[:idx_in_batch]), np.array(batch_y[:idx_in_batch])
    
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


