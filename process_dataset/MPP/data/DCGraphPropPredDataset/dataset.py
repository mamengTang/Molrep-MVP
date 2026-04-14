from torch_geometric.data import InMemoryDataset
import shutil, os
import os.path as osp
import torch
import re
from torch_sparse import SparseTensor

import numpy as np
from tqdm import tqdm
from ...utils.graph import smiles2graphwithface
from rdkit import Chem
from copy import deepcopy
from .deepchem_dataloader import (
    load_molnet_dataset,
    get_task_type,
)
from copy import deepcopy
import codecs
from subword_nmt.apply_bpe import BPE
import pandas as pd
from ....MPP.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data


class DGData(Data):
    def __cat_dim__(self, key, value, *args, **kwargs):
        if isinstance(value, SparseTensor):
            return (0, 1)
        elif bool(re.search("(index|face)", key)):
            return -1
        elif bool(re.search("(nf_node|nf_ring|nei_tgt_mask)", key)):
            return -1
        return 0

    def __inc__(self, key, value, *args, **kwargs):
        if bool(re.search("(ring_index|nf_ring)", key)):
            return int(self.num_rings.item())
        elif bool(re.search("(index|face|nf_node)", key)):
            return self.num_nodes
        else:
            return 0

def drug2emb_encoder(smile):
    vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt"
    sub_csv = pd.read_csv("ESPF/subword_units_map_chembl_freq_1500.csv")

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    idx2word_d = sub_csv['index'].values
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    max_d = 50
    t1 = dbpe.process_line(smile).split()  # split
    try:
        i1 = np.asarray([words2idx_d[i] for i in t1])  # index
    except:
        i1 = np.array([0])

    l = len(i1)
    if l < max_d:
        i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_d - l))
    else:
        i = i1[:max_d]
        input_mask = [1] * max_d

    return i, np.asarray(input_mask)


def eespf_tokenize(smile,vocab_path = "ESPF/drug_codes_chembl_freq_1500.txt",subword_map_path = "ESPF/subword_units_map_chembl_freq_1500.csv"):

    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    sub_csv = pd.read_csv(subword_map_path)
    idx2word_d = sub_csv['index'].values  # 所有子结构（token）
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))  # 词到索引的映射

    tokenized_smiles = dbpe.process_line(smile).split()  # # BPE 处理后拆分成 token 列表

    try:
        token_ids = np.asarray([words2idx_d[token] for token in tokenized_smiles])  #
    except KeyError:
        token_ids = np.array([0])

    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        raise ValueError(f"无法解析 SMILES: {smile}")
   

    atom_count = mol.GetNumAtoms() 
    atom_substructure_mapping = [-1] * atom_count

    atom_idx = 0
    substructure_idx = 0
    smile_cursor = 0  

    for token in tokenized_smiles:
  
        token_pos = smile.find(token, smile_cursor)
        if token_pos == -1:
            substructure_idx += 1
            continue  

        sub_atom_count = sum(1 for c in token if c.isalpha()) 
        for i in range(sub_atom_count):
            if atom_idx < atom_count:
                atom_substructure_mapping[atom_idx] = substructure_idx
                atom_idx += 1  

        smile_cursor = token_pos + len(token)
        substructure_idx += 1  

    max_length = 50
    seq_length = len(token_ids)

    if seq_length < max_length:
        padded_tokens = np.pad(token_ids, (0, max_length - seq_length), 'constant', constant_values=0)
        attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
    else:
        padded_tokens = token_ids[:max_length]
        attention_mask = [1] * max_length

    return tokenized_smiles,padded_tokens, np.asarray(attention_mask), seq_length, atom_substructure_mapping


def parse_atomic_symbols(token):

    periodic_table = set([
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr"
    ])


    matches = re.findall(r'[A-Z][a-z]?', token)
    return [m for m in matches if m in periodic_table]

def espf_tokenize(smile, mol, vocab_path="./ESPF/drug_codes_chembl_freq_1500.txt", subword_map_path="./ESPF/subword_units_map_chembl_freq_1500.csv"):
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

 
    sub_csv = pd.read_csv(subword_map_path)
    idx2word_d = sub_csv['index'].values  
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d)))) 

    tokenized_smiles = dbpe.process_line(smile).split()  
    match_atoms = []
    match_atoms_cnt = []
    for token in tokenized_smiles:
        token_atoms=parse_atomic_symbols(token)
        match_atoms.append(token_atoms)
        match_atoms_cnt.append(len(token_atoms))
    current_match_atom_cnt = [0]*len(match_atoms_cnt)

    try:
        token_ids = np.asarray([words2idx_d[token] for token in tokenized_smiles])
    except KeyError:
        token_ids = np.array([0])


    mol_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_count = len(mol_atoms)
    atom_substructure_mapping = [-1] * atom_count 

    atom_idx = 0 
    temp_token_pos = 0

    for atom_idx in range(0, atom_count):
        flag = False 
        current_token_pos = temp_token_pos
        for token in tokenized_smiles[current_token_pos:]:  

            if current_match_atom_cnt[temp_token_pos] >= match_atoms_cnt[temp_token_pos]:
                temp_token_pos += 1
            else:
                token_atoms = match_atoms[temp_token_pos]
                if mol_atoms[atom_idx] in token_atoms:
                    atom_substructure_mapping[atom_idx] = temp_token_pos
                    current_match_atom_cnt[temp_token_pos] += 1
                    match_atoms[temp_token_pos].remove(mol_atoms[atom_idx])
                    flag = True  
                    break 
                temp_token_pos += 1 
        if not flag:
            temp_token_pos = current_token_pos 

    max_length = 50
    seq_length = len(token_ids)

    if seq_length < max_length:
        padded_tokens = np.pad(token_ids, (0, max_length - seq_length), 'constant', constant_values=0)
        attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
    else:
        padded_tokens = token_ids[:max_length]
        attention_mask = [1] * max_length

    return tokenized_smiles, padded_tokens, np.asarray(attention_mask), seq_length, atom_substructure_mapping


class DCGraphPropPredDataset(InMemoryDataset):
    def __init__(self, name, root="./dataset/data", transform=None, pre_transform=None):
        assert name.startswith("dc-")
        name = name[len("dc-") :]
        self.name = name
        self.dirname = f"{name}"
        self.original_root = root
        self.root = osp.join(root, self.dirname)
        print(self.root)
        super().__init__(self.root, transform, pre_transform)
        self.data, self.slices, self._num_tasks = torch.load(self.processed_paths[0])

    def get_idx_split(self):
        path = os.path.join(self.root, "split", "split_dict.pt")
        return torch.load(path)

    @property
    def task_type(self):
        return get_task_type(self.name)

    @property
    def eval_metric(self):
        return "rocauc" if "classification" in self.task_type else "mae"

    @property
    def num_tasks(self):
        return self._num_tasks

    @property
    def raw_file_names(self):
        return ["data.npz"]

    @property
    def processed_file_names(self):
        return "geometric_data_processed.pt"

    def download(self):
        pass

    def process(self):
        train_idx = []
        valid_idx = []
        test_idx = []
        data_list = []
        _, dfs, _ = load_molnet_dataset(self.name)

        num_tasks = len(dfs[0]["labels"].values[0])

        for insert_idx, df in zip([train_idx, valid_idx, test_idx], dfs):
            smiles_list = df["text"].values.tolist()
            labels_list = df["labels"].values.tolist()
            assert len(smiles_list) == len(labels_list)

            for smiles, labels in zip(smiles_list, labels_list):
                data = DGData()
                mol = Chem.MolFromSmiles(smiles)
                graph = smiles2graphwithface(mol)

                assert len(graph["edge_feat"]) == graph["edge_index"].shape[1]
                assert len(graph["node_feat"]) == graph["num_nodes"]

                data.__num_nodes__ = int(graph["num_nodes"])

                if "classification" in self.task_type:
                    data.y = torch.as_tensor(labels).view(1, -1).to(torch.long)
                else:
                    data.y = torch.as_tensor(labels).view(1, -1).to(torch.float32)
                # atoms
                atom_features_list = []
                for atom in mol.GetAtoms():
                    atom_features_list.append(atom_to_feature_vector(atom))
                x = np.array(atom_features_list, dtype=np.int64)
                # bonds
                edges_list = []
                edge_features_list = []
                for bond in mol.GetBonds():
                    i = bond.GetBeginAtomIdx()
                    j = bond.GetEndAtomIdx()

                    edge_feature = bond_to_feature_vector(bond)

                    # add edges in both directions
                    edges_list.append((i, j))
                    edge_features_list.append(edge_feature)
                    edges_list.append((j, i))
                    edge_features_list.append(edge_feature)

                edge_index = np.array(edges_list, dtype=np.int64).T
                edge_attr = np.array(edge_features_list, dtype=np.int64)
                
                
                data.x = torch.from_numpy(x).to(torch.int64)
                data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
                data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)
                
                data.smiles_ori = smiles
                espf_smiles, data.tokens, data.attention_mask, substructure_num, data.atom2substructure = espf_tokenize(smiles,mol)



                data_list.append(data)
                insert_idx.append(len(data_list))
                data_list.append(data)

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        print("Saving...")
        torch.save((data, slices, num_tasks), self.processed_paths[0])

        os.makedirs(osp.join(self.root, "split"), exist_ok=True)
        torch.save(
            {
                "train": torch.as_tensor(train_idx, dtype=torch.long),
                "valid": torch.as_tensor(valid_idx, dtype=torch.long),
                "test": torch.as_tensor(test_idx, dtype=torch.long),
            },
            osp.join(self.root, "split", "split_dict.pt"),
        )
qm9_header_to_target = {
    "Alpha": 1,
    "Gap": 4,
    "HOMO": 2,
    "LUMO": 3,
    "Mu": 0,
    "Cv": 11,
    "G298": 10,
    "H298": 9,
    "R2": 5,
    "U298": 8,
    "U0": 7,
    "Zpve": 6,
    "Avg": None,  # 平均指标，不对应QM9的单一target
}

