from torch.utils.data import Dataset


class StepBatchDataset(Dataset):
    def __init__(self, data_size: int, **samples):
        self.data = samples
        self.data_length = data_size

    def __getitem__(self, index):
        sample = {}
        for k, v in self.data.items():
            if k == 'batch_size':
                continue
            sample[k] = v[index - 1]
        return sample

    def __len__(self):
        return self.data_length
