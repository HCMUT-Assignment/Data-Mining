from torch.utils.data import Dataset, DataLoader

class TSDataset(Dataset):
    
    def __init__(self, data, target):
        super(TSDataset, self).__init__()
        
        if data.shape[0] != target.shape[0]:
            raise ValueError("Size is not compatible")
        
        self.data   = data
        self.target = target
    
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        return self.data[index], self.target[index]
    
def get_data_loader(dataset: list, batch_size: int, shuffle: bool):

    data, target    = dataset
    TSdataset       = TSDataset(data, target)
    return DataLoader(TSdataset, batch_size=batch_size, shuffle = shuffle)

    
