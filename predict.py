import sys
import torch
from termcolor import colored
from smiles_embedding import build_vocab_util
sys.modules["build_vocab_util"] = build_vocab_util
from smiles_embedding.build_vocab_util import WordVocab
from MCPdataset import load_ind_data
from Mymodel import DrugTargetModel
from until import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(new_model, path_pretrain_model):
    pretrained_dict = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    new_model_dict = new_model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(pretrained_dict)
    new_model.load_state_dict(new_model_dict)
    return new_model


if __name__ == '__main__':
    # Evaluating the Results on the Independent Test Set of iRNA-ac4C
    file = 'data/ind_test.csv'
    vocab = WordVocab.load_vocab(f'data/BioSNAP_vocab.pkl')
    test_iter = load_ind_data(file,vocab)
    path_pretrain_model = "Model/MSF_DTI_BioSNAP_model.pt"
    model = DrugTargetModel().to(device)
    model = load_model(model, path_pretrain_model)
    model.eval()
    with torch.no_grad():
        for batch in test_iter:
            X_hot, smile_mask, graph, kbert, protein_seq, protein_mask, protT5, labels = batch
            X_hot = X_hot.to(device)
            smile_mask = smile_mask.to(device)
            graph = graph.to(device)
            kbert = kbert.to(device)
            protein_seq = protein_seq.to(device)
            protein_mask = protein_mask.to(device)
            protT5 = protT5.to(device)
            labels = labels.to(device)

        ind_performance, ind_roc_data, ind_prc_data, _ = evaluate(test_iter, model)
    ind_results = '\n' + '=' * 16 + colored(' Independent Test Set Performance', 'red') + '=' * 16 \
                  + '\n[ACC,\tAUROC,\t\tAUPRC,\t\tF1,\tSensitivity-Recall]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        ind_performance[0],ind_performance[1], ind_performance[2], ind_performance[3], ind_performance[4]) + '\n' + '=' * 60
    print(ind_results)

