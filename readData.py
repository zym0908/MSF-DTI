import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)
from rdkit.Chem import Draw
from rdkit.Chem import AllChem, rdMolDescriptors
import pandas as pd
import numpy as np
import torch.nn.functional as F
from rdkit import Chem
import networkx as nx
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, smiles_to_bigraph

def all_data_seq(protein, csv_path="data/BioSNAP_prott5_protein1200.csv", max_len = 1200):
    token2index = {
                    'X': 0,
                    'A': 1, 'R': 2, 'N': 3, 'D': 4, 'C': 5,
                    'Q': 6, 'E': 7, 'G': 8, 'H': 9, 'I': 10,
                    'L': 11, 'K': 12, 'M': 13, 'F': 14, 'P': 15,
                    'S': 16, 'T': 17, 'W': 18, 'Y': 19, 'V': 20,'U':21}
    token_list = []
    mask_list = []
    padded_seqs = []
    emb_list = []
    df_protT5 = pd.read_csv(csv_path)
    feat_cols = [f"emb_{i}" for i in range(1024)]
    protT5_dict = {}
    for _, row in df_protT5.iterrows():
        seq_key = row["protein"].strip()
        emb = row[feat_cols].values.astype(np.float32)
        protT5_dict[seq_key] = emb

    for seq in protein:
        L = len(seq)
        if L < max_len:
            pad_num = max_len - L
            padded_seq = seq + 'X' * pad_num
            mask = [True] * L + [False] * pad_num
        else:
            padded_seq = seq[:max_len]
            mask = [True] * max_len
        padded_seqs.append(padded_seq)
        mask_list.append(mask)

        seq_clean = seq.strip()
        if seq_clean not in protT5_dict:
            raise KeyError(f"protT52.csv lacks the protein sequence：{seq_clean}")
        emb_list.append(protT5_dict[seq_clean])

    for seq in padded_seqs:
        seq_id = [token2index[aa] for aa in seq]
        token_list.append(torch.tensor(seq_id))

    protein_seq = torch.stack(token_list, dim=0)  
    seq_mask = torch.tensor(mask_list, dtype=torch.bool)
    protT5_np = np.stack(emb_list, axis=0)
    protT5 = torch.from_numpy(protT5_np).float()

    return protein_seq, seq_mask, protT5

node_featurizer = CanonicalAtomFeaturizer()
edge_featurizer = CanonicalBondFeaturizer(self_loop=True)

def all_data_smile(smiles, labels, vocab, seq_len=200, kbert_csv_path="data/BioSNAP_drug_smiles_k_bert.csv"):

    list_X = []
    list_X_hot = []
    graph_list = [] 
    mask_list = [] 

    df_kbert = pd.read_csv(kbert_csv_path)
    feat_cols = [f"pretrain_feature_{i}" for i in range(1, 769)]
    kbert_map = {}
    for _, row in df_kbert.iterrows():
        smi_key = row["smiles"]
        vals = row[feat_cols].values

        clean_vals = []
        for v in vals:
            if isinstance(v, str):
                s = v.strip()
                if s in ("########", "#N/A", "#VALUE!", "", "nan"):
                    clean_vals.append(np.nan)
                else:
                    clean_vals.append(float(s))
            else:
                clean_vals.append(v)
        feat_arr = np.array(clean_vals, dtype=np.float32)
        feat_arr = np.nan_to_num(feat_arr, nan=0.0)
        kbert_map[smi_key] = feat_arr

    for item in smiles:
        content = [vocab.stoi.get(token, vocab.unk_index) for token in item]
        X_seq = [vocab.sos_index] + content + [vocab.eos_index]
        if len(X_seq) > seq_len:
            X_seq = X_seq[:seq_len - 1]
            X_seq[-1] = vocab.eos_index
            valid_raw_len = seq_len - 1
        else:
            valid_raw_len = len(X_seq)

        pad_num = seq_len - len(X_seq)
        padding = [vocab.pad_index] * pad_num
        X_seq.extend(padding)

        single_mask = [False] * valid_raw_len + [True] * pad_num
        mask_list.append(single_mask)

        X_tensor = torch.tensor(X_seq)
        X_hot_tensor = F.one_hot(X_tensor, len(vocab)).float()

        list_X.append(X_tensor)
        list_X_hot.append(X_hot_tensor)

        g = smiles_to_bigraph(
            item,
            add_self_loop=True,
            node_featurizer=node_featurizer,
            edge_featurizer=edge_featurizer,
            canonical_atom_order=False,
        )
        graph_list.append(g)

    kbert_feat_list = []
    for smi in smiles:
        feat = kbert_map[smi]
        kbert_feat_list.append(feat)
    k_bert = torch.from_numpy(np.array(kbert_feat_list)).float() 
    X = torch.stack(list_X, dim=0)  # [N, seq_len]
    X_hot = torch.stack(list_X_hot, dim=0)  # [N, seq_len, vocab_size]
    smile_seq_mask = torch.tensor(mask_list, dtype=torch.bool)

    return X, X_hot, smile_seq_mask, graph_list, k_bert, labels


