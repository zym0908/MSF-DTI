import sys
import dgl
import pandas as pd
from rdkit import RDLogger
from readData import all_data_smile, all_data_seq
from smiles_embedding import build_vocab_util
sys.modules["build_vocab_util"] = build_vocab_util
from smiles_embedding.build_vocab_util import WordVocab
RDLogger.DisableLog('rdApp.*')
import warnings
warnings.filterwarnings('ignore', category=PendingDeprecationWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import torch
from torch.utils.data import Dataset, DataLoader

class MCPDataset(Dataset):
    def __init__(self,smiles,protein,labels, vocab):
        super(MCPDataset, self).__init__()
        X, X_hot, smile_mask, graph_list,k_bert, labels = all_data_smile(smiles, labels, vocab)
        protein_seq, protein_mask,protT5 = all_data_seq(protein)
        self.X = X.long()
        self.X_hot = X_hot.float()
        self.smile_mask = smile_mask
        self.graph_list = graph_list
        self.k_bert = k_bert

        self.protein_seq = protein_seq.long()
        self.protein_mask = protein_mask
        self.protT5 = protT5

        self.labels = torch.LongTensor(labels)

    def __getitem__(self, idx):
        return (
            self.X[idx],
            self.X_hot[idx],
            self.smile_mask[idx],
            self.graph_list[idx],
            self.k_bert[idx],
            self.protein_seq[idx],
            self.protein_mask[idx],
            self.protT5[idx],
            self.labels[idx]
        )
    def __len__(self):
        return len(self.labels)

    def get_collate_fn(self):
        def _collate(data):
            _, X_hot, smile_mask, graph_list,k_bert, protein_seq, protein_mask,protT5, labels = map(list,zip(*data))
            X_hot = torch.stack(X_hot)
            smile_mask = torch.stack(smile_mask)
            graph = dgl.batch(graph_list)
            k_bert = torch.stack(k_bert)
            protein_seq = torch.stack(protein_seq)
            protein_mask = torch.stack(protein_mask)
            protT5 = torch.stack(protT5)
            labels = torch.stack(labels)
            return X_hot, smile_mask,graph,k_bert, protein_seq, protein_mask,protT5, labels
        return _collate

def construct_dataset(smiles,protein, labels, vocab, train=True, batch_size=64):
    dataset = MCPDataset(smiles, protein, labels, vocab)
    data_loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=train,
                            drop_last=False,
                            collate_fn=dataset.get_collate_fn())
    return data_loader

def load_ind_data(file,vocab):
    tmp = pd.read_csv(file)
    smile,protein, labels = tmp["smiles"].values.tolist(),tmp["protein"].values.tolist(), tmp["labels"].values.tolist()
    data_iter = construct_dataset(smile,protein, labels, vocab, train=False)
    return data_iter

if __name__ == '__main__':
    file = 'data/BioSNAP.csv'
    vocab = WordVocab.load_vocab(f'data/BioSNAP_vocab.pkl')
    df = pd.read_csv(file)
    smiles = df['smiles']
    protein = df['protein']
    labels = df['labels']

    dataset = MCPDataset(smiles, protein, labels, vocab)
    print("Total sample size：", len(dataset))

    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=dataset.get_collate_fn()
    )

    for batch in loader:
        X_hot, smile_mask, batched_graph, protein_seq, protein_mask, batch_labels = batch
        print("=" * 50)
        print(f"X_hot shape: {X_hot.shape}")
        print(f"smile_mask shape: {smile_mask.shape}")
        print(f"Total number of nodes in the batch graph: {batched_graph.num_nodes()}")
        print(f"protein_seq shape: {protein_seq.shape}")
        print(f"protein_mask shape: {protein_mask.shape}")
        print(f"labels shape: {batch_labels.shape}, labels dtype: {batch_labels.dtype}")
        print("=" * 50)
        break

    '''
    X_hot shape: torch.Size([8, 200, 83])
    smile_mask shape: torch.Size([8, 200])
    Total number of nodes in the batch graph: 166
    k_bert shape: torch.Size([8, 768])
    protein_seq shape: torch.Size([8, 1000])
    protein_mask shape: torch.Size([8, 1000])
    protT5 shape: torch.Size([8, 1280])
    labels shape: torch.Size([8]), labels dtype: torch.int64
    '''
