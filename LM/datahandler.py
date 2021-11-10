import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LMDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).to(device)
        self.y = torch.tensor(y).to(device)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
