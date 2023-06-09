import torch
import numpy as np
import pandas as pd

from torch.utils.data import Dataset


class DCNNDataset(Dataset):
    def __init__(self, file_path, window_size):
        self.win_size = window_size
        self._preprocess(file_path)
        
    def _preprocess(self, file_path):
        def convert(data):
            result = np.zeros(29)
            data = np.array(list(map(int, bin(int(data, 16))[2:])))
            result[-data.shape[0]:] = data
            return result
        
        df = pd.read_csv(file_path)
        self.data = df['Arbitration_ID'].map(convert).to_numpy()
        self.label = df['Class'].map(lambda x: 0 if x == 'Normal' else 1).to_numpy()

    def __len__(self):
        return self.data.shape[0] - self.win_size + 1
    
    def __getitem__(self, idx):
        x = torch.zeros((29, 29), dtype=np.float64)
        x[:self.win_size, :] = np.stack(self.data[idx:idx+self.win_size])
        y = torch.tensor(np.min([1, self.label[idx:idx+29].sum()])).to(torch.float32)
        return x, y


class LSTMDataset(Dataset):
    def __init__(self, file_path, window_size):
        self.data, self.label = [], []
        self.win_size = window_size
        self._preprocess(file_path)
        
    def _preprocess(self, file_path):
        def convert_id(data):
            result = np.zeros(11)
            data = np.array(list(map(int, bin(int(data, 16))[2:])))
            result[-data.shape[0]:] = data
            return result
        
        def convert_dlc(data):
            ret = np.zeros(8)
            if data > 0:
                ret[data - 1] = 1
            return ret
        
        def convert_data(data):
            ret = np.zeros(64) - 1
            for i, d in enumerate(data.split()):
                d = list(map(int, bin(int(d, 16))[2:]))
                d = [0] * (8 - len(d)) + d
                d = np.array(d)
                ret[i*8:i*8+8] = d            
            return ret
        
        df = pd.read_csv(file_path)
        id = df['Arbitration_ID'].map(convert_id).to_numpy()
        dlc = df['DLC'].map(convert_dlc).to_numpy()
        data = df['Data'].map(convert_data).to_numpy()
        self.data = [np.concatenate((i, d, da), axis=0) for i, d, da in zip(id, dlc, data)]
        self.label = df['Class'].map(lambda x: 0 if x == 'Normal' else 1).to_numpy()
        del df

    def __len__(self):
        return self.data.shape[0] - self.win_size + 1
    
    def __getitem__(self, idx):
        x = np.stack(self.data[idx:idx+self.win_size], dtype=np.float64)
        y = torch.tensor(min(1, self.label[idx:idx+self.win_size].sum()))
        return x, y