import torch
import torch.nn as nn
import torch.nn.functional as F

from protein_model import protein_model
from smiles_model import smiles_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MSC_Bi(nn.Module):
    def __init__(self, dim, num_heads=4, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0.4, proj_drop_ratio=0.4,
                 k1=3, k2=3):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.head_dim ** -0.5

        self.q_x = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_x = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.q_y = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_y = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_y = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

        self.k1 = k1
        self.k2 = k2

        self.alpha1 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.alpha2 = nn.Parameter(torch.tensor([0.5]), requires_grad=True)

        self.avgpool1 = nn.AvgPool1d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool1d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool1d(kernel_size=kernel[2], stride=s[2], padding=pad[2])
        self.norm_x = nn.LayerNorm(dim)
        self.norm_y = nn.LayerNorm(dim)

    def _multi_scale_enhance(self, feat):
        B, L, C = feat.shape
        f_t = feat.transpose(1,2)
        f1 = self.avgpool1(f_t)
        f2 = self.avgpool2(f_t)
        f3 = self.avgpool3(f_t)
        f_sum = f1 + f2 + f3
        out = f_sum.transpose(1,2)
        return out

    def _sparse_topk_attn(self, q, k, v):
        attn = (q @ k.transpose(-2, -1)) * self.scale
        Nq, Nk = q.size(-2), k.size(-2)

        k1_val = max(1, Nk // self.k1)
        k2_val = max(1, Nk // self.k2)

        # branch‑1
        mask1 = torch.zeros_like(attn, dtype=torch.bool)
        idx1 = torch.topk(attn, k=k1_val, dim=-1, largest=True).indices
        mask1.scatter_(-1, idx1, True)
        attn1 = attn.masked_fill(~mask1, -1e9)
        attn1 = F.softmax(attn1, dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = attn1 @ v

        # branch‑2
        mask2 = torch.zeros_like(attn, dtype=torch.bool)
        idx2 = torch.topk(attn, k=k2_val, dim=-1).indices
        mask2.scatter_(-1, idx2, True)
        attn2 = attn.masked_fill(~mask2, -1e9)
        attn2 = F.softmax(attn2, dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = attn2 @ v

        out = out1 * self.alpha1 + out2 * self.alpha2
        return out

    def forward(self, x, y):
        B, Nx, C = x.shape
        _, Ny, _ = y.shape

        y_enh = self._multi_scale_enhance(y)
        y_enh = self.norm_y(y_enh)

        x_enh = self._multi_scale_enhance(x)
        x_enh = self.norm_x(x_enh)

        kv_y = self.kv_y(y_enh)
        kv_y = kv_y.reshape(B, Ny, 2, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        k_y, v_y = kv_y[0], kv_y[1]

        q_x = self.q_x(x).reshape(B, Nx, self.num_heads, self.head_dim).permute(0,2,1,3)
        out_x_heads = self._sparse_topk_attn(q_x, k_y, v_y)
        x_out = out_x_heads.transpose(1,2).reshape(B, Nx, C)
        x_out = self.proj_drop(self.proj_x(x_out))

        kv_x = self.kv_x(x_enh)
        kv_x = kv_x.reshape(B, Nx, 2, self.num_heads, self.head_dim).permute(2,0,3,1,4)
        k_x, v_x = kv_x[0], kv_x[1]

        q_y = self.q_y(y).reshape(B, Ny, self.num_heads, self.head_dim).permute(0,2,1,3)
        out_y_heads = self._sparse_topk_attn(q_y, k_x, v_x)
        y_out = out_y_heads.transpose(1,2).reshape(B, Ny, C)
        y_out = self.proj_drop(self.proj_y(y_out))
        return x_out, y_out
        
class DrugTargetModel(nn.Module):
    def __init__(self,fusion_hidden=512,dropout=0.4,out_dim=2):
        super().__init__()
        self.device = device
        self.smiles_branch = smiles_model()
        self.protein_branch = protein_model()
        self.msc_bi = MSC_Bi(dim=256, num_heads=4)

        self.fusion_mlp = nn.Sequential(
            nn.Linear(256 * 4, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, fusion_hidden // 2),
            nn.LayerNorm(fusion_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 2, fusion_hidden // 4),
            nn.LayerNorm(fusion_hidden // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden // 4, out_dim)
        )

    def forward(self,graph,kbert, protein_seq,protT5):
        output1 = self.smiles_branch(graph, kbert)
        output2 = self.protein_branch(protein_seq,protT5)
        
        d_att, p_att = self.msc_bi(output1, output2)

        g1 = d_att.mean(dim=1)
        g2 = p_att.mean(dim=1)
        g3 = output1.mean(dim=1)
        g4 = output2.mean(dim=1)
        fuse_feat = torch.cat([g1, g2, g3, g4], dim=-1)
        out = self.fusion_mlp(fuse_feat)
        return out
