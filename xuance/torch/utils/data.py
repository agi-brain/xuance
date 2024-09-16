from torch.utils.data import Dataset


class StepBatchDataset(Dataset):
    def __init__(self, data_size: int, **samples):
        self.data = samples
        self.data_length = data_size

    def __getitem__(self, index):
        obs_batch = self.data['obs'][index-1]
        act_batch = self.data['actions'][index-1]
        next_batch = self.data['obs_next'][index-1]
        rew_batch = self.data['rewards'][index-1]
        ter_batch = self.data['terminals'][index-1]
        return obs_batch, act_batch, next_batch, rew_batch, ter_batch

    def __len__(self):
        return self.data_length
