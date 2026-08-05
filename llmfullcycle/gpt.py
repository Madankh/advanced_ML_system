import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

config = {
    vocab_size:int=50688,
    seq_len:int = 4096,
    n_head:int = 12,
    n_layer:int  6,
    n_embed:int = 768,
    n_kv_head:int = 6,
    window_pattern:str="SSSL"
}

class Linear(nn.Module):
    def __init__(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def norm(x):
    return F.rms_norm(x, (x.size(-1)))


def apply_rotation(x, cos_sin):
    assert x.ndim() == 4
    cos, sin = cos_sin
    d = x.shape[-1]//2
    x1,x2 = x[..., :d], x[..., d:]
    rotatex1 = x1 * cos + x2 * sin
    rotatex2 = x1 * (-sin) + x2 * cos
    return torch.cat([rotatex1, rotatex2], dim=-1)

def has_ve(layer_idx, n_layers):
    return layer_idx % 2 == (n_layers - 1) % 2

class Attention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_layer = config.n_layer
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.n_kv_head = config.n_kv_head
        self.head_dim = self.n_embed // self.n_kv_head
        self.wq = Linear(self.n_embed, self.n_head * self.head_dim, bias=False)
        self.wk = Linear(self.n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.wv = Linear(self.n_embed, self.n_kv_head * self.head_dim, bias=False)
        self.c = Linear(self.n_embed, self.n_embed, bias=False)
        self.ve_gate_channels = 12 
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x,ve, cos_sin, window_size, kv_cache):
        B,T,C = x.size()
        self.q = self.wq(x).view(B,T,self.n_head, self.head_dim)
        self.k = self.wk(x).view(B,T,self.n_kv_head, self.head_dim)
        self.v = self.wv(x).view(B,T,self.n_kv_head, self.head_dim)
        
        if ve is not None:
            ve = ve.view(B,T,self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q,k =  apply_rotation(q, cos_sin), apply_rotation(k, cos_sin)
        q , k = norm(q), norm(k)
        q = q * 1.2
        k = k * 1.2

        # Flash attention 
        if kv_cache is None:
            y = flash_attn.flash_attn_func(q,k,v causal=True, window_size=window_size)
        else:
            pass
        y = y.contiguous().view(B,T,-1)
        y = self.c(y)
        return y


class MLP(nn.Module):
    def __init__(self , config):
        super().__init__()
        self.n_embed = config.n_embed
        self.layer1 = Linear(self.n_embed, 4 * self.n_embed, bias=False)
        self.layer2 = Linear(self.n_embed * 4, self.n_embed, bias=False)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x).square()
        x = self.layer2(x)

        return x

class BLOCK(nn.Module):
    def __init__(self, config, layer_idx):
        self.n_embed = config.n_embed
        self.layer_idx = layer_idx
        self.mlp = MLP(config)
        self.attention = Attention(config, layer_idx)
    
    def forward(self, x,ve,cos_sin, window_size, kv_cache):
        x = x + self.attention(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x
    

class GPT(nn.Module):
    def __init__(self, config, pad_vocab_to=64):
        super().__init__()
        self.config = config
        self.window_size = self._compute_window_size(config)
        self.paddle_vocab_size = ((config.vocab_size + pad_vocab_to - 1) // pad_vocab_to) * pad_vocab_to
        
        self.transformer = nn.ModuleDict({
            "wte":nn.Embedding(self.paddle_vocab_size, self.config.n_embed),
            "h": nn.ModuleList([BLOCK(config, layer_idx) for layer_idx in range(config.n_layer)])
        })
        self.lm_head = Linear(self.config.n_embed, self.paddle_vocab_size)

        self.resid_lambdas = nn.Parameters(torch.ones(config.n_layer))
        self.x0_lambdas = nn.Parameters(torch.zeros(config.n_layer))

        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameters(torch.zeros(1))

        self.backout_lambda = nn.Parameters(0.2 * torch.ones(1))

        self.head_dim = self.config.n_embed // self.config.n_head
        self.kv_head = self.config.n_kv_head * self.head_dim
        self.value_embedding = nn.ModuleDict({
            str(i): nn.Embedding(self.paddle_vocab_size, self.kv_head)
            for i in range(self.config.n_layer)
            if has_ve(i, self.config.n_layer)
        })

        self.rotary_seq_len = self.config.seq_len * 10
        cos,sin = self._precompute_rotary_embedding(self.rotary_seq_len, self.head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.08),
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001),

        n_embed = self.config.n_embed
        s =  3 ** 0.5 * n_embed** -0.5
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attention.wq.weight,  -s, s)
            torch.nn.init.uniform_(block.attention.wk.weight,  -s, s)
            torch.nn.init.uniform_(block.attention.wv.weight, -s, s)
            torch.nn.init.zeros_(block.attention.c.weight)
            torch.nn.init.uniform_(block.mlp.layer1.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.layer2.weight)

        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))

        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        for ve in self.value_embedding.values():
            torch.nn.init.normal_(ve.weight, mean=0.0, std=0.08)

        for block in self.transformer.h:
            if block.attention.ve_gate is not None:
                torch.nn.init.normal_(block.attention.ve_gate.weight, mean=0.0, std=0.08)
        head_dim = self.config.n_embed // self.config.n_head
        cos, sin = self._precompute_rotary_embedding(self.rotary_seq_len, head_dim)


        


