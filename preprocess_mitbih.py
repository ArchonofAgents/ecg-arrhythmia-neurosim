import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

data = pd.read_csv('./data_root/ecg_segments_cleaned.csv', header=None)
labels = pd.read_csv('./data_root/labels.csv', header=None)

data_np = data.to_numpy()
labels_np = labels.to_numpy()

data_np = (data_np - data_np.min()) / (data_np.max() - data_np.min())  

data_tensor = torch.tensor(data_np, dtype=torch.float32)
labels_tensor = torch.tensor(labels_np, dtype=torch.long)

data_tensor = data_tensor.view(-1, 1, 180, 1)  # [batch_size, channels=1, height=180, width=1]

class MITBIHDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Save preprocessed tensors for reuse in training
torch.save(data_tensor, './data_root/data_tensor.pt')
torch.save(labels_tensor, './data_root/labels_tensor.pt')