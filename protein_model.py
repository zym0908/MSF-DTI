import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import AttentionalAggregation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Global_Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        gate_nn = nn.Linear(hidden_size, 1)
        self.at = AttentionalAggregation(gate_nn=gate_nn)

    def forward(self, x, batch):
        return self.at(x, batch)


class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model).to(device)
        self.pos_embed = nn.Embedding.from_pretrained(self.position_encoding(max_len, d_model), freeze=True).to(device)
        self.norm = nn.LayerNorm(d_model).to(device)

    def forward(self, x):
        seq_len = x.size(1)  # x: [batch_size, seq_len]
        pos = torch.arange(seq_len, dtype=torch.long, device=device)  # [seq_len]
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size, seq_len]
        return self.norm(self.tok_embed(x) + self.pos_embed(pos))

    @staticmethod
    def position_encoding(max_len, d_model):
        """
        Position encoding feature introduced in "Attention is all you need",
        the b is changed to 1000 for the short length of sequence.
        """
        b = 1000
        pos_encoding = np.zeros((max_len, d_model))
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pos_encoding[pos, i] = np.sin(pos / (b ** (2 * i / d_model)))
                if i + 1 < d_model:
                    pos_encoding[pos, i + 1] = np.cos(pos / (b ** (2 * i / d_model)))
        return torch.FloatTensor(pos_encoding)


class TokenProjector(nn.Module):
    def __init__(self, in_dim=1280, out_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.4)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        return self.fc(x)
        
class MSGA(nn.Module):
    def __init__(self, n_feats=128, gate_type="headwise"):
        super(MSGA, self).__init__()
        kernel_size_1 = 3  # 3-mer
        kernel_size_2 = 5  # 5-mer
        self.conv_3_1 = nn.Conv1d(n_feats, n_feats, kernel_size_1, padding=(kernel_size_1 // 2), bias=True)
        self.conv_3_2 = nn.Conv1d(n_feats * 2, n_feats * 2, kernel_size_1, padding=(kernel_size_1 // 2), bias=True)
        self.conv_5_1 = nn.Conv1d(n_feats, n_feats, kernel_size_2, padding=(kernel_size_2 // 2), bias=True)
        self.conv_5_2 = nn.Conv1d(n_feats * 2, n_feats * 2, kernel_size_2, padding=(kernel_size_2 // 2), bias=True)

        self.confusion = nn.Conv1d(n_feats * 4, n_feats*2, 1, padding=0, stride=1)
        self.shortcut_proj = nn.Conv1d(n_feats, n_feats*2, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.hidden_size = n_feats*2
        self.gate_type = gate_type

        self.attention_weights = nn.Linear(self.hidden_size, 1)
        if gate_type == "headwise":
            self.gate_proj = nn.Linear(self.hidden_size, 1)
        elif gate_type == "elementwise":
            self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size)
        elif gate_type is None:
            self.gate_proj = None
        else:
            raise ValueError(f"gate_type must be 'headwise', 'elementwise' or None, got {gate_type}")

    def forward(self, x):
        input_1 = x
        output_3_1 = self.relu(self.conv_3_1(input_1))
        output_5_1 = self.relu(self.conv_5_1(input_1))
        input_2 = torch.cat([output_3_1, output_5_1], dim=1)

        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        input_3 = torch.cat([output_3_2, output_5_2], dim=1)
        output = self.confusion(input_3)
        shortcut = self.shortcut_proj(x)
        output += shortcut

        output = output.permute(0, 2, 1)

        attn_weights = torch.softmax(self.attention_weights(output), dim=1)

        if self.gate_type is not None:
            gate_score = torch.sigmoid(
                self.gate_proj(output))
            weighted_output = output * gate_score * attn_weights
        else:
            weighted_output = output * attn_weights
        final_output = torch.sum(weighted_output, dim=1)
        return final_output

class protein_model(nn.Module):
    def __init__(self, emb_dim=128, hidden_size=256, max_len=1200):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_len = max_len
        self.embedding = Embedding(22, self.emb_dim, self.max_len)

        self.norm = nn.LayerNorm(hidden_size)
        self.p_proj = TokenProjector(1280, 256)
        self.msga = MSGA(n_feats=128, gate_type="headwise")
    def forward(self, protein_seq, protT5):
        emb = self.embedding(protein_seq)
        emb = emb.permute(0, 2, 1)
        emb = self.msga(emb)
        output1 = self.norm(emb)
        output1 = F.normalize(output1, p=2, dim=1)

        fuse = torch.cat([output1, protT5], dim=-1)
        output2 = self.p_proj(fuse)

        return output2


