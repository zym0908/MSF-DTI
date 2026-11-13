import os
import numpy as np
import pandas as pd
import torch
import time
import matplotlib.pyplot as plt
from termcolor import colored
from sklearn.model_selection import train_test_split
from Mymodel import DrugTargetModel
import sys
from MCPdataset import construct_dataset
from smiles_embedding import build_vocab_util
sys.modules["build_vocab_util"] = build_vocab_util
from smiles_embedding.build_vocab_util import WordVocab
from until import evaluate, reg_loss
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

seed = 42
seed_everything(seed)

os.makedirs("./Result-42", exist_ok=True)

class EarlyStopping:
    def __init__(self, patience=20, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = None

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        self.best_acc = val_acc
        path = 'best_network.pt'
        torch.save(model.state_dict(), path)

def train_test(train_iter, val_iter):
    global best_ROC
    net = DrugTargetModel().to(device)
    lr = 0.0001
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=15, delta=0.0001)

    best_acc = 0
    EPOCH = 200
    best_ROC = 0
    best_PRC = 0
    best_results = []
    best_performance = []

    for epoch in range(EPOCH):
        loss_ls = []
        t0 = time.time()

        net.train()
        for batch in train_iter:
            X_hot, smile_mask, graph, kbert, protein_seq, protein_mask, protT5, labels = batch
            X_hot = X_hot.to(device)
            smile_mask = smile_mask.to(device)
            graph = graph.to(device)
            kbert = kbert.to(device)
            protein_seq = protein_seq.to(device)
            protein_mask = protein_mask.to(device)
            protT5 = protT5.to(device)
            labels = labels.to(device)

            output = net(graph,kbert, protein_seq,protT5)
            loss = reg_loss(net, output, labels).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_ls.append(loss.item())

        net.eval()
        with torch.no_grad():
            train_performance, _, _, _ = evaluate(train_iter, net)
            val_performance, val_roc_data, val_prc_data, label_real = evaluate(val_iter, net)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 13 + ' Validation Performance. Epoch[{}] '.format(epoch + 1) + '=' * 13 \
                   + '\n[ACC,\tAUROC,\t\tAUPRC,\t\tF1,\tSensitivity-Recall]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            val_performance[0], val_performance[1], val_performance[2], val_performance[3],
            val_performance[4]) + '\n' + '=' * 60
        print(results)
        val_acc = val_performance[0]

        if val_acc > 0.70:
            filename = 'Model_ValACC[{:.4f}].pt'.format(val_acc)
            save_path_pt = os.path.join('./Result-42', filename)
            torch.save(net.state_dict(), save_path_pt, _use_new_zipfile_serialization=False)
        scheduler.step(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            best_performance = val_performance
            best_ROC = val_roc_data
            best_PRC = val_prc_data
            best_results = '\n' + '=' * 16 + colored(' Best Performance. Epoch[{}] ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tAUROC,\t\tAUPRC,\t\tF1,\tSensitivity-Recall]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}' \
                               .format(best_performance[0], best_performance[1], best_performance[2],
                                       best_performance[3], best_performance[4]) \
                           + '\n' + '=' * 60
        early_stopping(val_acc, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    best_ROC = np.array(best_ROC, dtype=object)
    best_PRC = np.array(best_PRC, dtype=object)

    roc_filename = 'ROC_AUROC[{:.4f}]_ACC[{:.4f}].npy'.format(best_performance[1], best_acc)
    full_path_roc = os.path.join('./Result-42', roc_filename)
    np.save(full_path_roc, best_ROC)

    prc_filename = 'PRC_AUPRC[{:.4f}]_ACC[{:.4f}].npy'.format(best_performance[2], best_acc)
    full_path_prc = os.path.join('./Result-42', prc_filename)
    np.save(full_path_prc, best_PRC)

    print(best_results)
    return best_performance, best_results, best_ROC, best_PRC

def train_fixed_split(train_file,val_file,vocab):
    train_df = pd.read_csv(train_file)
    train_smiles = train_df['smiles']
    train_protein = train_df["protein"]
    train_labels = train_df['labels']
    train_smiles, train_protein, train_labels = np.array(train_smiles), np.array(train_protein), np.array(train_labels)

    val_df = pd.read_csv(val_file)
    val_smiles = val_df['smiles']
    val_protein = val_df["protein"]
    val_labels = val_df['labels']
    val_smiles, val_protein, val_labels = np.array(val_smiles), np.array(val_protein), np.array(val_labels)

    train_iter = construct_dataset(train_smiles, train_protein, train_labels, vocab, train=True)
    val_iter = construct_dataset(val_smiles, val_protein, val_labels, vocab, train=False)

    best_performance, best_results, ROC, PRC = train_test(train_iter, val_iter)

    print('\n' + '=' * 16 + colored(' Final Performance ', 'red') + '=' * 16)
    print('[ACC,\tAUROC,\t\tAUPRC,\t\tF1,\tSensitivity-Recall]')
    print('{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        best_performance[0], best_performance[1], best_performance[2],
        best_performance[3], best_performance[4]
    ))
    print('=' * 60)
    return best_performance, best_results

if __name__ == '__main__':
    train_file = 'data/train.csv'
    val_file = 'data/val_test.csv'
    vocab = WordVocab.load_vocab(f'data/BioSNAP_vocab.pkl')
    best_performance, best_results = train_fixed_split(
        train_file,
        val_file,
        vocab
    )
