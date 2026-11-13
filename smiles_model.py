import torch
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear, Sequential, BatchNorm2d, ReLU, AdaptiveAvgPool2d
from torch import nn
from torch.nn import functional as F
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch_geometric.nn import AttentionalAggregation
from dgl import function as fn 
import dgl
from dgllife.model import GAT, GCN
# from dgllife.utils import MaxPooling
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DrugTokenizer(nn.Module):
    def __init__(self, in_dim=768, out_dim=256, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.out_dim = out_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, out_dim * num_tokens),
            nn.LayerNorm(out_dim * num_tokens),
            nn.GELU(),
            nn.Dropout(0.4)
        )
    def forward(self, x):
        # (B,768)
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)
        out = self.fc(x) # (B,768) -> (B,2048)
        return out.view(-1, self.num_tokens, self.out_dim)  # (B,2048) -> (B,8,256)
class MaxPooling(nn.Module):
    def forward(self, g, feat):
        g.ndata["h"] = feat
        return dgl.max_nodes(g, "h")

class GINConvDGL(nn.Module):
    def __init__(self, in_dim, out_dim, epsilon=0.0, activation=nn.LeakyReLU()):
        super().__init__()
        self.eps = nn.Parameter(torch.tensor(epsilon))
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            activation,
            nn.Linear(out_dim, out_dim)
        )
    def forward(self, g, feat):
        with g.local_scope():
            g.ndata['h'] = feat
            g.update_all(dgl.function.copy_u('h', 'm'), dgl.function.sum('m', 'neigh'))
            neigh = g.ndata['neigh']
            x = (1 + self.eps) * feat + neigh
            return self.mlp(x)
class GATv2ConvDGL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        negative_slope: float = 0.2,
        feat_drop: float = 0.0,
        attn_drop: float = 0.0,
        add_self_loop: bool = True,
        share_weights: bool = False
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.negative_slope = negative_slope
        self.add_self_loop = add_self_loop
        self.share_weights = share_weights

        self.lin_l = nn.Linear(in_dim, num_heads * out_dim, bias=False)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = nn.Linear(in_dim, num_heads * out_dim, bias=False)

        self.att = nn.Parameter(torch.empty(1, num_heads, out_dim))

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)

        self.bias = nn.Parameter(torch.empty(num_heads * out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_l.weight)
        nn.init.xavier_uniform_(self.lin_r.weight)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor):
        N = feat.shape[0]
        with g.local_scope():
            if self.add_self_loop:
                g = dgl.add_self_loop(g)

            feat = self.feat_drop(feat)
            h_l = self.lin_l(feat).view(N, self.num_heads, self.out_dim)
            if self.share_weights:
                h_r = h_l
            else:
                h_r = self.lin_r(feat).view(N, self.num_heads, self.out_dim)

            g.ndata["h_l"] = h_l
            g.ndata["h_r"] = h_r

            g.apply_edges(fn.u_add_v("h_r", "h_l", "h_sum"))
            h_sum = g.edata.pop("h_sum")
            h_sum = F.leaky_relu(h_sum, negative_slope=self.negative_slope)

            e = (h_sum * self.att).sum(dim=-1)
            g.edata["e"] = e

            g.edata["alpha"] = dgl.ops.edge_softmax(g, g.edata["e"])
            g.edata["alpha"] = self.attn_drop(g.edata["alpha"])
            g.edata["alpha"] = g.edata["alpha"].unsqueeze(-1)

            g.update_all(
                fn.u_mul_e("h_r", "alpha", "m"),
                fn.sum("m", "out")
            )
            out = g.ndata["out"]
            out = out.reshape(N, self.num_heads * self.out_dim)
            out = out + self.bias
            return out

class GCNGATGIN(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 num_heads,
                 feat_drops,
                 attn_drops,
                 alphas,
                 residuals,
                 agg_modes,
                 activations):
        super().__init__()
        h1, h2_per_head, h3 = hidden_feats
        self.gcn1 = GCN(in_feats=in_feats, hidden_feats=[h1])

        self.gatv2 = GATv2ConvDGL(
            in_dim=h1,
            out_dim=h2_per_head,
            num_heads=num_heads,
            feat_drop=feat_drops[0],
            attn_drop=attn_drops[0],
            negative_slope=alphas[0],
            add_self_loop=True
        )

        gat_out_dim = num_heads * h2_per_head

        self.gin = GINConvDGL(in_dim=gat_out_dim, out_dim=h3, activation=nn.LeakyReLU())

    def forward(self, g, feats):
        x = self.gcn1(g, feats)
        x = self.gatv2(g, x)
        x = self.gin(g, x)
        return x

class smiles_model(nn.Module):
    def __init__(self, input_dim = 83, hidden_size = 256, num_head = 4, n_layers = 2, dropout = 0.4, device = device):
        super(smiles_model, self).__init__()
        self.in_size = input_dim
        self.hidden_size = hidden_size
        self.device = device
        self.d_proj = DrugTokenizer(1024,  256, 8)
        self.gnn_layers = GCNGATGIN(
            in_feats=74,
            hidden_feats=[128, 128, 256],
            num_heads=4,
            feat_drops=[0.4],
            attn_drops=[0.4],
            alphas=[0.2],
            residuals=[False],
            agg_modes=["sum"],
            activations=[nn.LeakyReLU()]
        )
        self.pool = MaxPooling()
        self.graph_proj = nn.Sequential(Linear(128, hidden_size), ReLU(), nn.Dropout(dropout))
    def forward(self, graph,kbert):
        atom_feats = self.gnn_layers(graph, graph.ndata["h"])
        graph_vec = self.pool(graph, atom_feats)
        fuse = torch.cat([graph_vec, kbert], dim=-1)
        output2 = self.d_proj(fuse)
        return output2
