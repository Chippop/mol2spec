import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os
import sys
sys.path.append("/home/code/mol2sepc")
class SmilesPeaksDataset(Dataset):
    def __init__(self, smiles, peaks_feature) -> None:
        
        self.peaks_feature = peaks_feature
        self.smiles_feature = smiles

        assert len(self.peaks_feature) == len(self.smiles_feature)
        
    def __getitem__(self, index):
        return self.smiles_feature[index], self.peaks_feature[index]
    
    def __len__(self):
        return len(self.smiles_feature)
    