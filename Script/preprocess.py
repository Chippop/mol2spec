import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import pandas as pd 
import numpy as np 
import pickle
import lmdb
import os
import glob
import torch
from multiprocessing import Pool
import re
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing
from rdkit import RDLogger
from rdkit import rdBase
import numpy as np
from openbabel import openbabel
from collections import Counter
rdBase.DisableLog('rdApp.error')
rdkit_logger = RDLogger.logger()
rdkit_logger.setLevel(RDLogger.ERROR)
import logging

# Disable RDKit warnings and Python warnings
RDLogger.DisableLog('rdApp.*') 
warnings.filterwarnings(action='ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def inner_smi2coords(smi, mode='fast', remove_hs=True):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:  # Check if RDKit was able to parse the SMILES string
                return None, None
            mol = AllChem.AddHs(mol)
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            if len(atoms) == 0:
                return None, None
            res = AllChem.EmbedMolecule(mol, randomSeed=24)
            if res == 0 or (res == -1 and mode == 'heavy'):
                try:
                    AllChem.MMFFOptimizeMolecule(mol)
                except:
                    pass
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)
            else:
                AllChem.Compute2DCoords(mol)
                coordinates = mol.GetConformer().GetPositions().astype(np.float32)

            if remove_hs:
                idx = [i for i, atom in enumerate(atoms) if atom != 'H']
                atoms = [atom for i, atom in enumerate(atoms) if i in idx]
                coordinates = coordinates[idx]

            return atoms, coordinates
        except Exception as e:
            logger.error(f"Failed to process SMILES: {smi} with error: {e}")
            return None, None

def xyz2smi(atoms, coords):
    mol = openbabel.OBMol()
    for j in range(len(coords)):
        atom = mol.NewAtom()
        atom.SetAtomicNum(openbabel.GetAtomicNum(atoms[j]))
        x, y, z = map(float, coords[j])
        atom.SetVector(x, y, z)
    mol.ConnectTheDots()
    mol.PerceiveBondOrders()
    obConversion = openbabel.OBConversion()
    obConversion.SetOutFormat('smi')
    smi = obConversion.WriteString(mol)
    return smi.split('\t\n')[0]

def normalize_atoms(atom):
    return re.sub("\d+", "", atom)

def smi_tokenizer(smi):
    """
    Tokenize a SMILES molecule or reaction
    """
    pattern =  "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    try:
        tokens = [token for token in regex.findall(smi)]
        assert smi == ''.join(tokens)
        return tokens
    except Exception as e:
        print(f"Error tokenizing SMILES {smi}. Error: {e}")
        return []

def identify_outlier_bounds(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)
    return lower_bound, upper_bound



def sort_directories(dir_names):
    # Extract numbers from directory names and sort by these numbers
    sorted_dirs = sorted(dir_names, key=lambda x: int(x.split('_')[-1]))
    return sorted_dirs


def process_data(args):
    i, _idx, coord_list, atom_list, e_abs, e_emi, reorg_e = args
    _coord = np.array(coord_list)# np.array(coord_list.tolist(), dtype=np.float32)
    _atom = [normalize_atoms(a) for a in np.array(atom_list)]

    assert len(_coord) == len(_atom), 'length of coord and atom should be the same'
    _smi = xyz2smi(_atom, _coord)
    _atom, _coord = inner_smi2coords(_smi)    
    if _atom is None or _coord is None:
            return None  
    _smi_tokens = smi_tokenizer(_smi)
    dd = pickle.dumps({
        'ID': i,
        'coordinates': [_coord],
        'atoms': _atom,
        'atoms_charge': None,
        'smi': _smi,
        'e_abs': e_abs,
        'target' : (e_abs, e_emi, reorg_e),
        'e_emi': e_emi,
        'reorg_e': reorg_e,
        'num_atoms': len(_atom),
        'smi_tokens': _smi_tokens,
    }, protocol=-1)
    return f'{i}'.encode("ascii"), dd

def check_keys_and_merge(data1, data2, data3):
    data1 = np.load(data1, allow_pickle=True)
    data2 = np.load(data2, allow_pickle=True)
    data3 = np.load(data3, allow_pickle=True)
    
    keys1 = set(data1.keys())
    keys2 = set(data2.keys())
    keys3 = set(data3.keys())
    
    if keys1 != keys2 or keys2 != keys3:
        print("Error: The keys in the npz files are not the same.")
        print(f"Keys in data1: {keys1}")
        print(f"Keys in data2: {keys2}")
        print(f"Keys in data3: {keys3}")
        return None

    merged_data = {key: np.concatenate((data1[key], data2[key], data3[key])) for key in keys1}
    
    return merged_data

# abs吸收能，emi发射能，reorg重组能
def writelmdb():
    # outpath = '/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/rdkit_dft'
    # # data1 = np.load('/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/oled_1000.npz', allow_pickle=True)
    # # data2 = "/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/OOMs.npz"
    # # data3 = "/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/opt.npz"
    # data1 = '/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/oled_1000.npz'
    # data2 = '/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/OOMs.npz'
    # data3 = '/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/opt.npz'
    
    data = check_keys_and_merge(data1, data2, data3)
    coord = data['coord']
    symbol = data['symbol']
    
    e_abs = data['e_abs']
    e_emi = data['e_emi']
    reorg_e = data['reorg_e']
    
    print("shape: \n", )
    print(coord.shape)
    print(symbol.shape)
    print(e_abs.shape)
    print(e_emi.shape)
    print(reorg_e.shape)

    print(f"e_abs max:{max(e_abs)}, min: {min(e_abs)}")
    print(f"e_emi max:{max(e_emi)}, min: {min(e_emi)}")
    print(f"reorg_e max:{max(reorg_e)}, min: {min(reorg_e)}")



    np.random.seed(42)
    idx = np.random.permutation(range(len(symbol)))
    _, val_ratio = 0.8, 0.2
    val_idx = idx[:int(len(symbol)*val_ratio)]
    train_idx = idx[int(len(symbol)*val_ratio):]

    nthreads = multiprocessing.cpu_count()
    print("Number of CPU cores:", nthreads)

    for name, idx in [('valid.lmdb', val_idx),  
                      ('train.lmdb', train_idx)]:
        outputfilename = os.path.join(outpath, name)
        try:
            os.remove(outputfilename)
        except:
            pass
        env_new = lmdb.open(
            outputfilename,
            subdir=False,
            readonly=False,
            lock=False,
            readahead=False,
            meminit=False,
            max_readers=1,
            map_size=int(1024*1024*1024*40),
        )
        txn_write = env_new.begin(write=True)
        with Pool(nthreads) as pool:
            i = 0
            args_list = [(i, _idx, coord[_idx], symbol[_idx],  e_abs[_idx], e_emi[_idx], reorg_e[_idx]) for i, _idx in enumerate(tqdm(idx))]
            for result in pool.imap(process_data, args_list):
                # 检查process_data函数的结果，如果不是None，则继续处理
                if result is not None:
                    key, inner_output = result
                    txn_write.put(key, inner_output)
                    i += 1
                    if i % 1000 == 0:
                        txn_write.commit()
                        txn_write = env_new.begin(write=True)
            # for key, inner_output in tqdm(pool.imap(process_data, args_list), total=len(idx)):
            #     txn_write.put(key, inner_output)
            #     i += 1
            #     if i % 1000 == 0:
            #         txn_write.commit()
            #         txn_write = env_new.begin(write=True)
            print('{} process {} lines'.format(name, i))
            txn_write.commit()
            env_new.close()       



def token_collect():
    dir_path = '/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/rdkit_dft'
    atoms_collects = []
    counts = []
    smi_tokens_list = []
    for lmdb_name in ['train.lmdb', 'valid.lmdb']:
        outputfilename = os.path.join(dir_path, lmdb_name)
        env = lmdb.open(
                outputfilename,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(10e9),
            )
        txn = env.begin()
        _keys = list(txn.cursor().iternext(values=False))
        for idx in tqdm(range(len(_keys))):
            datapoint_pickled = txn.get(f'{idx}'.encode("ascii"))
            if datapoint_pickled == None:
                continue
            data = pickle.loads(datapoint_pickled)
            atoms_collects.extend(data['atoms'])
            smi_tokens_list.extend(data['smi_tokens'])
            counts.append(len(data['atoms']))
        env.close()
    from collections import Counter

    # Calculate the length of each atom list using a list comprehension

    # Use Counter to count the frequencies of each length
    length_frequencies = Counter(counts)

    print(length_frequencies)
    print(max(length_frequencies))
    atoms_collects = pd.Series(atoms_collects).value_counts()

    tokens = ['[PAD]', '[CLS]', '[SEP]', '[UNK]']
    token_dict = {token: 0 for token in tokens}
    token_series = pd.Series(token_dict)
    
    # Concatenate the token series at the beginning of the atom collects
    combined_series = pd.concat([token_series, atoms_collects])
    combined_series.to_csv('/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/rdkit_dft/atoms_counts.csv', index=True, header=False)
    with open('/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/rdkit_dft/dict.txt', 'w') as file:
        for item in combined_series.index.tolist():
            file.write(item + '\n')

    smi_dict_collects = pd.Series(smi_tokens_list).value_counts()
    smi_dict_series = pd.concat([token_series, smi_dict_collects])
    with open('/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/rdkit_dft/smi.dict.txt', 'w') as file:
            for item in smi_dict_series.index.tolist():
                file.write(item + '\n')
    smi_dict_series.to_csv('/vepfs/fs_projects/FunMG/OOM_Mol_Design/classifier_specific_data/ooms_opt/rdkit_dft/smi_atoms_counts.csv',index=True, header=False)

    num_atoms = []
    e_abs_list = []
    e_emi_list = []
    reorg_e_list = []
    train_smi_list = []
    for lmdb_name in ['train.lmdb']:
        outputfilename = os.path.join(dir_path, lmdb_name)
        env = lmdb.open(
                outputfilename,
                subdir=False,
                readonly=False,
                lock=False,
                readahead=False,
                meminit=False,
                max_readers=1,
                map_size=int(10e9),
            )
        txn = env.begin()
        _keys = list(txn.cursor().iternext(values=False))
        for idx in tqdm(range(len(_keys))):
            datapoint_pickled = txn.get(f'{idx}'.encode("ascii"))
            if datapoint_pickled is not None:
                data = pickle.loads(datapoint_pickled)
                e_abs_list.append(data['e_abs'])
                e_emi_list.append(data['e_emi'])
                reorg_e_list.append(data['reorg_e'])
                num_atoms.append(data['num_atoms'])
                train_smi_list.append(data['smi'])
        env.close()

    properties = {
                                'e_abs': e_abs_list,
                                'e_emi': e_emi_list,
                                'reorg_e':reorg_e_list,
                }
    # # e_abs, e_emi, reorg_e
    assert len(e_abs_list) == len(e_emi_list)
    mean_values = []
    std_values = []
    property_names = []
    property_norms = {'num_atoms': torch.tensor(num_atoms, dtype=torch.int64)}
    for property_key in properties:
        values = properties[property_key]
        values = torch.tensor(values, dtype=torch.float32)
        mean = torch.mean(values)
        std = torch.tensor(values.std(0), dtype=torch.float32)
        property_norms[property_key] = {}
        property_norms[property_key]['mean'] = mean
        property_norms[property_key]['std'] = std
        property_norms[property_key]['data'] = values
        mean_values.append(mean.item())
        std_values.append(std.item())
        property_names.append(property_key)

    print(property_norms)
    with open('property_norms.pkl', 'wb') as f:
        pickle.dump(property_norms, f)
    formatted_properties = {
        "data": {
            "mean": mean_values,
            "std": std_values,
            "target_name": property_names
        }
    }
    print(formatted_properties)



if __name__ == '__main__':
    writelmdb()
    token_collect()


# {'data': {'mean': [0.09780015796422958, 0.06646512448787689, 0.031335026025772095], 'std': [0.027580518275499344, 0.03385605290532112, 0.022875284776091576], 'target_name': ['e_abs', 'e_emi', 'reorg_e']}}