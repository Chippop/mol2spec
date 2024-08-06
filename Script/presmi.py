

import sys
sys.path.append("/vepfs/fs_users/yftc/code/mol2sepc")
import numpy as np
import pandas as pd
from module.unimol_tools.data.conformer import ConformerGen, coords2unimol, inner_smi2coords
from module.unimol_tools.data.dictionary import Dictionary
from module.unimol_tools.weights.weighthub import WEIGHT_DIR
import os

def formatsmi(smi_list):
    smiles_formatList = []
    CG = ConformerGen()
    for smiles in smi_list:
        atoms, coordinates = inner_smi2coords(smi = smiles, seed = 42, mode = "fast", remove_hs = True)
        dictionary = Dictionary.load(os.path.join(WEIGHT_DIR, CG.dict_name))
        dictionary.add_symbol("[MASK]", is_special=True)
        reaslut = coords2unimol(atoms=atoms, coordinates=coordinates, dictionary=dictionary)
        smiles_formatList.append(reaslut)
        # print(reaslut["src_distance"])
        # print(type(reaslut["src_distance"]))
        import numpy as np
        import torch
        import torch.nn as nn
        from module.TransfomerEncoder import _TransformerEncode
        from module.unimol_tools.models import UniMolModel
        src_tokens = torch.tensor(reaslut["src_tokens"], dtype=torch.long).to(device="cuda")
        src_distance = torch.tensor(reaslut["src_distance"], dtype=torch.float32).to(device="cuda")
        src_coord = torch.tensor(reaslut["src_coord"], dtype=torch.float32).to(device="cuda")
        src_edge_type = torch.tensor(reaslut["src_edge_type"], dtype=torch.long).to(device="cuda")
        # print(src_coord)
        # print(type(src_coord))
        Un = UniMolModel(output_dim=1, data_type="molecule")
        smi_embd = Un(src_tokens,src_distance,src_coord,src_edge_type)
        print(smi_embd)


        break
    return smiles_formatList



data_path = "/vepfs/fs_users/yftc/code/mol2sepc/data/train.pkl"
df = pd.read_pickle(data_path)
# smiles_list = ["C1=CC=C(C=C1)C=O","CCCCC/C(=C/C1=CC=CC=C1)/C=O"]
smiles_feature = df["cleaned_smiles"].values
smiles_formatList = formatsmi(smiles_feature)
import numpy as np
a = np.array(smiles_formatList)
np.save("/vepfs/fs_users/yftc/code/mol2sepc/data/1111111.npy",a)

# file = "/vepfs/fs_users/yftc/code/mol2sepc/data/1111.npy"
# test = np.load(file,allow_pickle=True)
# print(len(test))
