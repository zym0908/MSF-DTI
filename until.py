from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
import numpy as np
import torch
import torch.nn as nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate(data_iter, net):
    pred_prob = []
    label_pred = []
    label_real = []
    with torch.no_grad():
        for batch in data_iter:
            X_hot, smile_mask, graph,kbert, protein_seq, protein_mask,protT5, labels = batch

            X_hot = X_hot.to(device)
            smile_mask = smile_mask.to(device)
            graph = graph.to(device)
            kbert = kbert.to(device)
            protein_seq = protein_seq.to(device)
            protein_mask = protein_mask.to(device)
            protT5 = protT5.to(device)
            labels = labels.to(device)

            outputs = net(graph,kbert, protein_seq,protT5)
            pred_prob_positive = outputs[:, 1]
            pred_prob = pred_prob + pred_prob_positive.tolist()
            label_pred = label_pred + outputs.argmax(dim=1).tolist()
            label_real = label_real + labels.tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    return performance, roc_data, prc_data, label_real

def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1
    # Accuracy
    ACC = float(tp + tn) / test_num
    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)
    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)
    # Precision
    if tp + fp == 0:
        Precision = 0
    else:
        Precision = float(tp) / (tp + fp)
    # F1分数
    if Precision + Recall == 0:
        F1 = 0
    else:
        F1 = 2 * (Precision * Recall) / (Precision + Recall)
    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / (np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)
    AUC = auc(FPR, TPR)
    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)
    performance = [ACC, AUC, AP, F1, Recall]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    l2_lambda = 0.01
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.norm(param, p=2)
    total_loss = criterion(output, label) + l2_lambda * regularization_loss
    return total_loss
