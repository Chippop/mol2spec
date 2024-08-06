import torch
import pandas as pd
import numpy as np
from torch import optim
import os
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
sys.path.append("/home/code/mol2spec")
# from code.mol2sepc.module.Multimodel import Multimodel

from Script.Dataset import SmilesPeaksDataset
from Script.datanorm import PeakProcessor
from module.Multimodel import Multimodel
from module.unimol_tools.weights import WEIGHT_DIR
from module.unimol_tools.data import Dictionary
from module.unimol_tools.utils.util import pad_1d_tokens, pad_2d, pad_coords
MODEL_CONFIG = {
    "weight":{
        "protein": "poc_pre_220816.pt",
        "molecule_no_h": "mol_pre_no_h_220816.pt",
        "molecule_all_h": "mol_pre_all_h_220816.pt",
        "crystal": "mp_all_h_230313.pt",
        "oled": "oled_pre_no_h_230101.pt",
    },
    "dict":{
        "protein": "poc.dict.txt",
        "molecule_no_h": "mol.dict.txt",
        "molecule_all_h": "mol.dict.txt",
        "crystal": "mp.dict.txt",
        "oled": "oled.dict.txt",
    },
}
def contrastive_loss(smiles_embedding, peaks_embedding):
    cosine_sim = nn.functional.cosine_similarity(smiles_embedding, peaks_embedding)
    loss = 1 - cosine_sim.mean()
    return loss

def batch_collate_fn(samples):
        remove_hs = False
        name = "no_h" if remove_hs else "all_h" 
        name = "molecule" + '_' + name
        dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, MODEL_CONFIG['dict'][name]))
        padding_idx = dictionary.pad()

        batch = {}
        for k in samples[0][0].keys():
            if k == 'src_coord':
                v = pad_coords([torch.tensor(s[0][k],dtype=torch.float32).to(device) for s in samples], pad_idx=0.0)
            elif k == 'src_edge_type':
                v = pad_2d([torch.tensor(s[0][k],dtype=torch.long).to(device) for s in samples], pad_idx=padding_idx)
            elif k == 'src_distance':
                v = pad_2d([torch.tensor(s[0][k],dtype=torch.float32).to(device) for s in samples], pad_idx=0.0)
            elif k == 'src_tokens':
                v = pad_1d_tokens([torch.tensor(s[0][k],dtype=torch.long).to(device) for s in samples], pad_idx=padding_idx)
            batch[k] = v
        try:
            label = torch.tensor([s[1] for s in samples],dtype=torch.float32).to(device)
        except:
            label = None
        return batch, label




if __name__ == "__main__":
    data_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/test.pkl"
    smiles_feature_path = "/vepfs/fs_users/yftc/code/mol2spec_git/data/test_smi_format.npy"#this file need Script/presmi.py
    mass_range = 1000
    bins = 200
    input_dim = 1
    embd_dim = 64
    num_head = 8
    ff_dim = 128
    num_layer = 2

    lr = 0.01
    num_epochs = 50

    peakp = PeakProcessor(mass_range, bins ,data_path)
    np.set_printoptions(threshold=np.inf)
    peaks_feature = peakp.get_peaks_feature()
    smiles_feature = np.load(smiles_feature_path, allow_pickle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Multimodel(input_dim=input_dim, embd_dim=embd_dim, num_head=num_head, ff_dim=ff_dim, num_layer=num_layer).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = SmilesPeaksDataset(smiles_feature, peaks_feature)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True,collate_fn=batch_collate_fn)

    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for smiles, peaks in tqdm(dataloader):
            smiles, peaks = smiles,peaks.to(device)
            optimizer.zero_grad()
            smiles_embd,peaks_embd = model(smiles,peaks)
            loss = contrastive_loss(smiles_embd, peaks_embd)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

        

