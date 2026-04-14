import os
import os.path as osp
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import re
import random
from ogb.utils.torch_util import replace_numpy_with_torchtensor
from ogb.utils.url import decide_download, download_url, extract_zip
from rdkit import Chem
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
import warnings
warnings.filterwarnings('error')
from ogb.utils.features import  (atom_to_feature_vector, bond_to_feature_vector)
import codecs
from subword_nmt.apply_bpe import BPE
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit import Chem
from rdkit import Chem
from rdkit.Chem import AllChem
from ogb.utils.features import (atom_to_feature_vector, bond_to_feature_vector)
import ast
import string

def parse_atomic_symbols(token):
    periodic_table = set([  
        "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
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

    def norm(sym: str) -> str:
    
        sym = sym.strip()
        if sym in ("se", "as"):
            return sym.title()  
        if len(sym) == 1:
            return sym.upper()
        return sym[0].upper() + sym[1:].lower()

    out = []

 
    aromatic_syms = []
    for match in re.finditer(r"[A-Z][a-z]?", token):
        symbol = match.group(0)
        start = match.start()
        end = match.end()
        aromatic_syms.append((symbol, start))
    normal_syms2 = []
    for i in aromatic_syms:
        if i[0] not in periodic_table:
            aaaaaa = re.findall(r"[A-Z]?", i[0])
            aaaaaa = [m for m in aaaaaa if m != '']
            assert len(aaaaaa) == 1
            normal_syms2.extend([(aaaaaa[0],i[1])])
        else:
            pass
    out.extend(normal_syms2)

  
    aromatic_syms = []
    for match in re.finditer(r"(se|as|[bcnops])", token):
        symbol = match.group(0)
        start = match.start()
        aromatic_syms.append((symbol, start))
    out.extend(aromatic_syms)
 
    out = [item for item in out if item[0] not in (None, '', [])]
    num = [pos for sym, pos in sorted(out, key=lambda x: x[1])]
    num_set = set(num)

   
    upper_outside = [(token[i],i) for i in range(len(token))
                    if i not in num_set and token[i] in string.ascii_uppercase]
    out.extend(upper_outside)

    out = [sym for sym, pos in sorted(out, key=lambda x: x[1])]

    out = [norm(x) for x in out]

   
    return [x for x in out if x in periodic_table]

def build_atom_to_token_map(order, match_atoms_cnt):
    n_atoms = max(order) + 1
    mapping = [-1] * n_atoms
    p = 0
    for t, c in enumerate(match_atoms_cnt):
        for _ in range(c):
            a = order[p]
            mapping[a] = t
            p += 1
    return mapping



def espf_tokenize(smile, mol, vocab_path="./ESPF/drug_codes_chembl_freq_1500.txt", subword_map_path="./ESPF/subword_units_map_chembl_freq_1500.csv"):
    bpe_codes_drug = codecs.open(vocab_path)
    dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
    bpe_codes_drug.close()

    sub_csv = pd.read_csv(subword_map_path)
    idx2word_d = sub_csv['index'].values  # 所有子结构（token）
    words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))

    tokenized_smiles = dbpe.process_line(smile).split()
    match_atoms = []
    match_atoms_cnt = []
    for nn,token in enumerate(tokenized_smiles):
        token_atoms=parse_atomic_symbols(token)
        match_atoms.extend(token_atoms)
        match_atoms_cnt.append(len(token_atoms))

    try:
        token_ids = np.asarray([words2idx_d[token] for token in tokenized_smiles])
    except KeyError:
        token_ids = np.array([0])
    
    mol_atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atom_count = len(mol_atoms)
    atom_substructure_mapping = [-1] * atom_count
    order = ast.literal_eval(mol.GetProp('_smilesAtomOutputOrder'))
    if len(match_atoms) != len(mol_atoms) or  len(mol_atoms) != len(order) or len(match_atoms) != len(order):
        return None, None, None, None, None

    atom_substructure_mapping = build_atom_to_token_map(order, match_atoms_cnt)

    max_length = 50
    seq_length = len(token_ids)

    if seq_length < max_length:
        padded_tokens = np.pad(token_ids, (0, max_length - seq_length), 'constant', constant_values=0)
        attention_mask = [1] * seq_length + [0] * (max_length - seq_length)
    else:
        padded_tokens = token_ids[:max_length]
        attention_mask = [1] * max_length

    return tokenized_smiles, padded_tokens, np.asarray(attention_mask), seq_length, atom_substructure_mapping

def gen_confs_rank_by_mmff(smiles,mol_from_sdf):

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("SMILES error")

    try:
        AllChem.EmbedMultipleConfs(
                        mol,
                        numConfs=5,          
                        randomSeed=42,       
                        numThreads=0,        
                        useRandomCoords=True 
                    )
    except RuntimeError as e:
        return None, None, None
    if mol.GetNumConformers() == 0:
        print(f"error")
        return None, None, None
    try:
        li = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=22222)
        if len(li) == 0:
            raise RuntimeError("error")
        
        # 获取优化后的能量并排序
        li = [t[1] for t in li]
        sortidx = torch.argsort(torch.tensor(li))

        minidx = int(sortidx[0])             
        min_energy = li[minidx]              

        pos = mol.GetConformer(minidx).GetPositions()

        atom_mapping = mol.GetSubstructMatch(mol_from_sdf)
        
        for i in range(len(pos)):
            assert mol.GetAtomWithIdx(i).GetSymbol() != 'H'


        return pos, min_energy,atom_mapping
    except RuntimeError as e:
        print(f"Error: {e}")
        return None, None, None 


class PCQM4Mv2Dataset(InMemoryDataset):
    def __init__(
            self,
            root="./dataset/",
            transform=None,
            pre_transform=None,
            xyzdir='./pcqm4m-v2/xyz',
            mask_ratio=0.5
    ):
        self.original_root = root
        self.mask_ratio = mask_ratio
        self.folder = osp.join(root, "pcqm4m-v2")
        self.version = 1

        self.url = "https://dgl-data.s3-accelerate.amazonaws.com/dataset/OGB-LSC/pcqm4m-v2.zip"

        if osp.isdir(self.folder) and (
                not osp.exists(osp.join(self.folder, f"RELEASE_v{self.version}.txt"))
        ):
            print("PCQM4Mv2 dataset has been updated.")
            if input("Will you update the dataset now? (y/N)\n").lower() == "y":
                shutil.rmtree(self.folder)

        self.xyzdir = xyzdir

        super().__init__(self.folder, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def raw_file_names(self):
        return "data.csv.gz"

    @property
    def processed_file_names(self):
        return "data.pt"

    def download(self):
        if decide_download(self.url):
            path = download_url(self.url, self.original_root)
            extract_zip(path, self.original_root)
            os.unlink(path)
        else:
            print("Stop download.")
            exit(-1)

    def process(self):
        Count_wrong = 0
        Count_wrong2 = 0
        data_df = pd.read_csv(osp.join(self.raw_dir, "data.csv.gz"))
        smiles_list = data_df["smiles"]
        homolumogap_list = data_df["homolumogap"]

        split_dict = self.get_idx_split()
        train_idxs = split_dict["train"].tolist()
        print("Converting SMILES strings into graphs...")
        data_list = []

        for i in tqdm(range(3378605)):
            # data = DGData()
            if "Ge" in smiles_list[i]:
                continue
            data = Data()
            smiles = smiles_list[i]
            homolumogap = homolumogap_list[i]

            prefix = i // 10000
            prefix = "{0:04d}0000_{0:04d}9999".format(prefix)
            xyzfn = osp.join(self.xyzdir, prefix, f"{i}.xyz")
            mol_from_sdf = next(SDMolSupplier(xyzfn))

            pos_from_sdf = mol_from_sdf.GetConformer(0).GetPositions()
            if Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False,canonical=True) != Chem.MolToSmiles(mol_from_sdf, isomericSmiles=False,canonical=True):
                continue
            smiles = Chem.MolToSmiles(mol_from_sdf, isomericSmiles=True)
            mol = Chem.MolFromSmiles(smiles)
            smiles = Chem.MolToSmiles(mol, canonical=True) 
            if len(Chem.GetMolFrags(mol)) > 1:
                continue
            atom_mapping = mol.GetSubstructMatch(mol_from_sdf)
            pos = np.zeros_like(pos_from_sdf)
            for sdf_idx, smiles_idx in enumerate(atom_mapping):
                pos[smiles_idx] = pos_from_sdf[sdf_idx]
            num_atoms = mol.GetNumAtoms()

            pos2,energy2,atom_mapping2 = gen_confs_rank_by_mmff(smiles,mol_from_sdf)
            
            if pos2 is None or len(pos2) == 0:
                print("MMFF defeat")
                pos2 = pos
                # energy2 = energy
                atom_mapping2 = atom_mapping
                Count_wrong2 += 1
            else:
                # assert energy != energy2
                if atom_mapping != atom_mapping2:
                    pos2 = pos
                    # energy2 = energy
                    atom_mapping2 = atom_mapping
                    print("Warning")
                    Count_wrong += 1
        
            atom_features_list = []
            for atom in mol.GetAtoms():
                atom_features_list.append(atom_to_feature_vector(atom))
            x = np.array(atom_features_list, dtype=np.int64)
         
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            edge_index = np.array(edges_list, dtype=np.int64).T
            edge_attr = np.array(edge_features_list, dtype=np.int64)

            data.x = torch.from_numpy(x).to(torch.int64)
            data.y = torch.Tensor([homolumogap])
            data.edge_index = torch.from_numpy(edge_index).to(torch.int64)
            data.edge_attr = torch.from_numpy(edge_attr).to(torch.int64)

            data.pos = torch.from_numpy(pos).to(torch.float)
            data.pos2 = torch.from_numpy(pos2).to(torch.float)

            target_row = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float)
            is_zero_row = torch.all(torch.from_numpy(pos).to(torch.float) == target_row, dim=1)
            is_zero_row2 = torch.all(torch.from_numpy(pos2).to(torch.float) == target_row, dim=1)
            if torch.any(is_zero_row) or torch.any(is_zero_row2):
                continue

            espf_smiles, data.tokens, data.attention_mask, substructure_num, data.atom2substructure = espf_tokenize(smiles,mol)
            if espf_smiles is None:
                continue
            data.ori_smiles = smiles

            if data.pos.size()[0] == 0 or data.pos.size()[1] == 0:
                print("zero!")
                print(data.pos.size())
                continue
            data.num_nodes = num_atoms

            data_list.append(data)

        data, slices = self.collate(data_list)
        print(Count_wrong)
        print(Count_wrong2)

        print("Saving...")
        torch.save((data, slices), "./pcqm4m-v2/processed/data.pt")

    def get_idx_split(self):
        split_dict = replace_numpy_with_torchtensor(
            torch.load(osp.join(self.root, "split_dict.pt"), weights_only=False)
        )
        return split_dict

if __name__ == "__main__":
    smiles = "c1cccc(C(=O)CCBc2ccccc2)c1"
    mol = Chem.MolFromSmiles(smiles)
    smiles = Chem.MolToSmiles(mol, canonical=False) 
    espf_tokenize(smiles,mol)

    