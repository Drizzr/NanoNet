


class Dataset():

    def __getitem__(self, index):
        raise NotImplementedError("Subclasses of Dataset should implement __getitem__.")