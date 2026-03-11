import torch.nn as nn
import os,sys
import torch as th
sys.path.append("/project/HARL-main/")
from harl.utils.models_tools import init, get_active_func, get_init_method
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import numpy as np
import math
"""MLP modules."""


class MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
        """
        super(MLPLayer, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        layers = [
            init_(nn.Linear(input_dim, hidden_sizes[0])),
            active_func,
            nn.LayerNorm(hidden_sizes[0]),
        ]

        for i in range(1, len(hidden_sizes)):
            layers += [
                init_(nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])),
                active_func,
                nn.LayerNorm(hidden_sizes[i]),
            ]

        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x)


class MLPBase(nn.Module):
    """A MLP base module."""

    def __init__(self, args, obs_shape):
        super(MLPBase, self).__init__()
        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]
        obs_dim = obs_shape[0]
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
        self.mlp = MLPLayer(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func)

    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x

class SelfAttnBlock(nn.Module):
    """Single self-attention + feedforward block for vector observations.
    Expects input shape [B, D]. We project to tokens (optionally multiple) via a linear 'tokenizer'.
    For simplicity, we treat the feature vector as a single token sequence of length n_tokens created by chunking.
    """
    def __init__(self, dim, n_heads=4, ff_ratio=4.0, dropout=0.0):
        super().__init__()
        self.dim = dim
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, int(ff_ratio * dim)),
            nn.GELU(),
            nn.Linear(int(ff_ratio * dim), dim),
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [B, T, D]
        h, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + h)
        h2 = self.ff(x)
        x = self.ln2(x + h2)
        return x

class SelfAttnMLPBase(nn.Module):
    """Self-attention enhanced base for policy/critic. Converts flat obs into patch tokens then applies transformer blocks.
    Args expected in args dict:
        self_attn_n_blocks: int number of attention blocks
        self_attn_n_heads: int number of attention heads
        self_attn_ff_ratio: float expansion in feedforward
        self_attn_dropout: float dropout (optional)
        hidden_sizes: list, last element used as embedding dim (must be consistent)
        use_feature_normalization: bool layernorm on input
    """
    def __init__(self, args, obs_shape):
        super().__init__()
        self.use_feature_normalization = args.get("use_feature_normalization", False)
        self.hidden_sizes = args["hidden_sizes"]
        embed_dim = self.hidden_sizes[0]
        obs_dim = obs_shape[0]
        self.input_proj = nn.Linear(obs_dim, embed_dim)
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
        n_blocks = args.get("self_attn_n_blocks", 2)
        n_heads = args.get("self_attn_n_heads", 4)
        ff_ratio = args.get("self_attn_ff_ratio", 4.0)
        dropout = args.get("self_attn_dropout", 0.0)
        self.blocks = nn.ModuleList([
            SelfAttnBlock(embed_dim, n_heads=n_heads, ff_ratio=ff_ratio, dropout=dropout)
            for _ in range(n_blocks)
        ])
        # Final MLP to reach last hidden size if different
        if len(self.hidden_sizes) > 1:
            mlp_layers = []
            in_dim = embed_dim
            for h in self.hidden_sizes[1:]:
                mlp_layers.append(nn.Linear(in_dim, h))
                mlp_layers.append(nn.ReLU())
                in_dim = h
            self.final_mlp = nn.Sequential(*mlp_layers)
            self.out_dim = self.hidden_sizes[-1]
        else:
            self.final_mlp = nn.Identity()
            self.out_dim = embed_dim

    def forward(self, x):
        # x: [B, obs_dim]
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        x = self.input_proj(x)  # [B, D]
        # treat as single token sequence length 1
        x = x.unsqueeze(1)  # [B,1,D]
        for blk in self.blocks:
            x = blk(x)
        x = x.squeeze(1)  # [B,D]
        x = self.final_mlp(x)
        return x

class Hypernet(nn.Module):
    def __init__(self, input_dim, hidden_dim, main_input_dim, main_output_dim, activation_func, n_heads):
        super(Hypernet, self).__init__()

        self.n_heads = n_heads
        # the output dim of the hypernet
        output_dim = main_input_dim * main_output_dim
        # the output of the hypernet will be reshaped to [main_input_dim, main_output_dim]
        self.main_input_dim = main_input_dim
        self.main_output_dim = main_output_dim

        self.multihead_nn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            get_active_func(activation_func),
            nn.Linear(hidden_dim, output_dim * self.n_heads),
            )

    def forward(self, x):
        # [...,  main_output_dim + main_output_dim + ... + main_output_dim]
        # [bs, main_input_dim, n_heads * main_output_dim]
        return self.multihead_nn(x).view([-1, self.main_input_dim, self.main_output_dim * self.n_heads])


class Merger(nn.Module):
    def __init__(self, head, fea_dim):
        super(Merger, self).__init__()
        self.head = head
        if head > 1:
            self.weight = Parameter(th.Tensor(1, head, fea_dim).fill_(1.))
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        """
        :param x: [bs, n_head, fea_dim]
        :return: [bs, fea_dim]
        """
        if self.head > 1:
            return th.sum(self.softmax(self.weight) * x, dim=1, keepdim=False)
        else:
            return th.squeeze(x, dim=1)

class MLPLayerUni2HPN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func,hpn_structure):
        """Initialize the MLP layer.
        Args:
            input_dim: (int) input dimension.
            hidden_sizes: (list) list of hidden layer sizes.
            initialization_method: (str) initialization method.
            activation_func: (str) activation function.
            hpn_structure [self.own_dim,[self.goal_num,self.goal_dim],[self.ally_num,self.ally_dim],self.n_heads,self.hpn_hyper_dim]
        """
        super(MLPLayerUni2HPN, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.own_dim = hpn_structure[0]
        self.goal_num = hpn_structure[1][0]
        self.goal_dim = hpn_structure[1][1]
        self.ally_num = hpn_structure[2][0]
        self.ally_dim = hpn_structure[2][1]
        self.n_heads = hpn_structure[3]
        self.hpn_hyper_dim = hpn_structure[4]
        self.hidden_sizes = hidden_sizes
        
        # Unique Features (do not need hyper net)
        self.fc1_own = init_(nn.Linear(self.own_dim, self.hidden_sizes[0], bias=True))  # only one bias is OK

        # %%%%%%%%%%%%%%%%%%%%%% Hypernet-based API input layer %%%%%%%%%%%%%%%%%%%%
        # Multiple entities (use hyper net to process these features to ensure permutation invariant)
        self.hyper_input_w_goal = Hypernet(
            input_dim=self.goal_dim, hidden_dim=self.hpn_hyper_dim,
            main_input_dim=self.goal_dim, main_output_dim=self.hidden_sizes[0],
            activation_func=activation_func, n_heads=self.n_heads
        )  # output shape: (enemy_feats_dim * self.hidden_sizes[0])
        self.hyper_input_w_ally = Hypernet(
            input_dim=self.ally_dim, hidden_dim=self.hpn_hyper_dim,
            main_input_dim=self.ally_dim, main_output_dim=self.hidden_sizes[0],
            activation_func=activation_func, n_heads=self.n_heads
        )  # output shape: ally_feats_dim * rnn_hidden_dim

        # self.unify_input_heads = nn.Linear(self.hidden_sizes[0] * self.n_heads, self.hidden_sizes[0])
        self.unify_input_heads = Merger(self.n_heads, self.hidden_sizes[0])

        self.fc2=nn.Sequential(*[get_active_func(activation_func),nn.LayerNorm(self.hidden_sizes[0]),
            init_(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
            get_active_func(activation_func),
            nn.LayerNorm(self.hidden_sizes[1]),
        ])
    def forward(self, x):
        # (1) Own feature
        own_feats_t,enemy_feats_t,ally_feats_t = x[:, :self.own_dim], x[:, self.own_dim:self.own_dim + self.goal_num * self.goal_dim].reshape(-1,self.goal_dim), x[:, self.own_dim + self.goal_num * self.goal_dim:].reshape(-1,self.ally_dim)
        bs = x.size(0)
        embedding_own = self.fc1_own(own_feats_t)  # [bs , rnn_hidden_dim]
        # (2) Enemy feature: [bs * n_enemies, enemy_fea_dim] -> [bs * n_enemies, enemy_feats_dim, rnn_hidden_dim * n_heads]
        input_w_enemy = self.hyper_input_w_goal(enemy_feats_t)
        # [bs * n_enemies, 1, enemy_fea_dim] * [bs * n_enemies, enemy_fea_dim, rnn_hidden_dim * head] = [bs * n_enemies, 1, rnn_hidden_dim * head]
        embedding_enemies = th.matmul(enemy_feats_t.unsqueeze(1), input_w_enemy).view(
            bs , self.goal_num, self.n_heads, self.hidden_sizes[0]
        )  # [bs, n_enemies, n_head, rnn_hidden_dim]
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs, n_head, rnn_hidden_dim]

        # (3) Ally features: [bs * n_allies, ally_fea_dim] -> [bs * n_allies, ally_feats_dim, rnn_hidden_dim * n_heads]
        input_w_ally = self.hyper_input_w_ally(ally_feats_t)
        # [bs * n_allies, 1, ally_fea_dim] * [bs * n_allies, ally_fea_dim, rnn_hidden_dim * head] = [bs * n_allies, 1, rnn_hidden_dim * head]
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), input_w_ally).view(
            bs, self.ally_num, self.n_heads, self.hidden_sizes[0]
        )  # [bs, n_allies, n_head, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs, n_head, rnn_hidden_dim]
        # Final embedding, merge multiple heads into one. -> [bs, n_head, rnn_hidden_dim]
        # (4) concatenate all features
        embedding = embedding_own + self.unify_input_heads(
            embedding_enemies + embedding_allies
        )
        # (5) MLP
        return self.fc2(embedding)

class MLPLayerUni2DeepSet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func,deepset_structure):
        super(MLPLayerUni2DeepSet, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.own_dim = deepset_structure[0]
        self.goal_num = deepset_structure[1][0]
        self.enemy_feats_dim = deepset_structure[1][1]
        self.ally_num = deepset_structure[2][0]
        self.ally_feats_dim = deepset_structure[2][1]
        self.hidden_sizes = hidden_sizes
        
        # Unique Features (do not need hyper net)
        self.fc1_own = init_(nn.Linear(self.own_dim, self.hidden_sizes[0], bias=True))  # only one bias is OK
        # Ally features
        self.fc1_ally = nn.Linear(self.ally_feats_dim, self.hidden_sizes[0], bias=False)  # only one bias is OK

        # Enemy features
        self.fc1_enemy = nn.Linear(self.enemy_feats_dim, self.hidden_sizes[0], bias=False)  # only one bias is OK
        self.fc2=nn.Sequential(*[get_active_func(activation_func),nn.LayerNorm(self.hidden_sizes[0]),
            init_(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
            get_active_func(activation_func),
            nn.LayerNorm(self.hidden_sizes[1]),
        ])
    def forward(self, x):
        # (1) Own feature
        own_feats_t,enemy_feats_t,ally_feats_t = x[:, :self.own_dim], x[:, self.own_dim:self.own_dim + self.goal_num * self.enemy_feats_dim].reshape(-1,self.enemy_feats_dim), x[:, self.own_dim + self.goal_num * self.enemy_feats_dim:].reshape(-1,self.ally_feats_dim)
        bs = x.size(0)
        
        # (1) Own feature
        embedding_own = self.fc1_own(own_feats_t.reshape(-1, self.own_dim))
        # (3) Enemy feature
        embedding_enemies = self.fc1_enemy(enemy_feats_t).view(
            bs , self.goal_num, self.hidden_sizes[0]
        )  
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]

        # (4) Ally features
        embedding_allies = self.fc1_ally(ally_feats_t).view(
            bs, self.ally_num, self.hidden_sizes[0]
        )  # [bs * n_agents, n_enemies, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]
        aggregated_embedding = embedding_own + embedding_enemies + embedding_allies  # [bs * n_agents, rnn_hidden_dim]
        
        # (5) MLP
        return self.fc2(aggregated_embedding)

class MLPLayerUni1DeepSet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func,deepset_structure):
        super(MLPLayerUni1DeepSet, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        self.own_dim = deepset_structure[0]
        self.goal_num = deepset_structure[1][0]
        self.enemy_feats_dim = deepset_structure[1][1]
        self.hidden_sizes = hidden_sizes
        # print(deepset_structure,hidden_sizes)
        
        # Unique Features (do not need hyper net)
        self.fc1_own = init_(nn.Linear(self.own_dim, self.hidden_sizes[0], bias=True))  # only one bias is OK
        # Enemy features
        self.fc1_enemy = nn.Linear(self.enemy_feats_dim, self.hidden_sizes[0], bias=False)  # only one bias is OK
        self.fc2=nn.Sequential(*[get_active_func(activation_func),nn.LayerNorm(self.hidden_sizes[0]),
            init_(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
            get_active_func(activation_func),
            nn.LayerNorm(self.hidden_sizes[1]),
        ])
    def forward(self, x):
        # (1) Own feature
        # print(x.shape)
        own_feats_t,enemy_feats_t = x[:, :self.own_dim], x[:, self.own_dim:self.own_dim + self.goal_num * self.enemy_feats_dim].reshape(-1,self.enemy_feats_dim)
        bs = x.size(0)
        embedding_own = self.fc1_own(own_feats_t.reshape(-1, self.own_dim))
        # (3) Enemy feature
        embedding_enemies = self.fc1_enemy(enemy_feats_t).view(
            bs , self.goal_num, self.hidden_sizes[0]
        )  
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]
        aggregated_embedding = embedding_own + embedding_enemies  # [bs * n_agents, rnn_hidden_dim]
        # (5) MLP
        return self.fc2(aggregated_embedding)
        
class MLPLayer2DeepSet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func,deepset_structure):
        super(MLPLayer2DeepSet, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.own_dim = deepset_structure[0]
        self.goal_num = deepset_structure[1][0]
        self.enemy_feats_dim = deepset_structure[1][1]
        self.ally_num = deepset_structure[2][0]
        self.ally_feats_dim = deepset_structure[2][1]
        self.hidden_sizes = hidden_sizes
        
        # Unique Features (do not need hyper net)
        # self.fc1_own = init_(nn.Linear(self.own_dim, self.hidden_sizes[0], bias=True))  # only one bias is OK
        # Ally features
        self.fc1_ally = nn.Linear(self.ally_feats_dim, self.hidden_sizes[0], bias=True)  # only one bias is OK

        # Enemy features
        self.fc1_enemy = nn.Linear(self.enemy_feats_dim, self.hidden_sizes[0], bias=False)  # only one bias is OK
        self.fc2=nn.Sequential(*[get_active_func(activation_func),nn.LayerNorm(self.hidden_sizes[0]),
            init_(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
            get_active_func(activation_func),
            nn.LayerNorm(self.hidden_sizes[1]),
        ])
    def forward(self, x):
        # (1) Own feature
        own_feats_t,enemy_feats_t,ally_feats_t = x[:, :self.own_dim], x[:, self.own_dim:self.own_dim + self.goal_num * self.enemy_feats_dim].reshape(-1,self.enemy_feats_dim), x[:, self.own_dim + self.goal_num * self.enemy_feats_dim:].reshape(-1,self.ally_feats_dim)
        bs = x.size(0)
        
        # (1) Own feature
        # embedding_own = self.fc1_own(own_feats_t.reshape(-1, self.own_dim))
        # (3) Enemy feature
        embedding_enemies = self.fc1_enemy(enemy_feats_t).view(
            bs , self.goal_num, self.hidden_sizes[0]
        )  
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]

        # (4) Ally features
        embedding_allies = self.fc1_ally(ally_feats_t).view(
            bs, self.ally_num, self.hidden_sizes[0]
        )  # [bs * n_agents, n_enemies, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]
        aggregated_embedding = embedding_enemies + embedding_allies  # [bs * n_agents, rnn_hidden_dim]
        
        # (5) MLP
        return self.fc2(aggregated_embedding)
    
class MLPLayer1DeepSet(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func,deepset_structure):
        super(MLPLayer1DeepSet, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.own_dim = deepset_structure[0]
        self.goal_num = deepset_structure[1][0]
        self.enemy_feats_dim = deepset_structure[1][1]
        self.ally_num = deepset_structure[2][0]
        self.ally_feats_dim = deepset_structure[2][1]
        self.hidden_sizes = hidden_sizes
        # Enemy features
        self.fc1_enemy = nn.Linear(self.enemy_feats_dim, self.hidden_sizes[0], bias=True)  # only one bias is OK
        self.fc2=nn.Sequential(*[get_active_func(activation_func),nn.LayerNorm(self.hidden_sizes[0]),
            init_(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
            get_active_func(activation_func),
            nn.LayerNorm(self.hidden_sizes[1]),
        ])
    def forward(self, x):
        # (1) Own feature
        own_feats_t,enemy_feats_t,ally_feats_t = x[:, :self.own_dim], x[:, self.own_dim:self.own_dim + self.goal_num * self.enemy_feats_dim].reshape(-1,self.enemy_feats_dim), x[:, self.own_dim + self.goal_num * self.enemy_feats_dim:].reshape(-1,self.ally_feats_dim)
        bs = x.size(0)
        # (3) Enemy feature
        embedding_enemies = self.fc1_enemy(enemy_feats_t).view(bs , self.goal_num, self.hidden_sizes[0])  
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs * n_agents, rnn_hidden_dim]
        aggregated_embedding = embedding_enemies  # [bs * n_agents, rnn_hidden_dim]
        # (5) MLP
        return self.fc2(aggregated_embedding)
    
    
class MLPLayer2HPN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func,hpn_structure):
        super(MLPLayer2HPN, self).__init__()
        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.own_dim = hpn_structure[0]
        self.goal_num = hpn_structure[1][0]
        self.goal_dim = hpn_structure[1][1]
        self.ally_num = hpn_structure[2][0]
        self.ally_dim = hpn_structure[2][1]
        self.n_heads = hpn_structure[3]
        self.hpn_hyper_dim = hpn_structure[4]
        self.hidden_sizes = hidden_sizes
        
        self.hyper_input_w_goal = Hypernet(
            input_dim=self.goal_dim, hidden_dim=self.hpn_hyper_dim,
            main_input_dim=self.goal_dim, main_output_dim=self.hidden_sizes[0],
            activation_func=activation_func, n_heads=self.n_heads
        )  # output shape: (enemy_feats_dim * self.hidden_sizes[0])
        self.hyper_input_b_goal = Hypernet(
            input_dim=self.goal_dim, hidden_dim=self.hpn_hyper_dim,
            main_input_dim=1, main_output_dim=self.hidden_sizes[0],
            activation_func=activation_func, n_heads=self.n_heads
        )  # output shape: (enemy_feats_dim * self.hidden_sizes[0])
        self.hyper_input_w_ally = Hypernet(
            input_dim=self.ally_dim, hidden_dim=self.hpn_hyper_dim,
            main_input_dim=self.ally_dim, main_output_dim=self.hidden_sizes[0],
            activation_func=activation_func, n_heads=self.n_heads
        )  # output shape: ally_feats_dim * rnn_hidden_dim

        # self.unify_input_heads = nn.Linear(self.hidden_sizes[0] * self.n_heads, self.hidden_sizes[0])
        self.unify_input_heads = Merger(self.n_heads, self.hidden_sizes[0])

        self.fc2=nn.Sequential(*[get_active_func(activation_func),nn.LayerNorm(self.hidden_sizes[0]),
            init_(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
            get_active_func(activation_func),
            nn.LayerNorm(self.hidden_sizes[1]),
        ])
    def forward(self, x):
        # (1) Own feature
        own_feats_t,enemy_feats_t,ally_feats_t = x[:, :self.own_dim], x[:, self.own_dim:self.own_dim + self.goal_num * self.goal_dim].reshape(-1,self.goal_dim), x[:, self.own_dim + self.goal_num * self.goal_dim:].reshape(-1,self.ally_dim)
        bs = x.size(0)
        # embedding_own = self.fc1_own(own_feats_t)  # [bs , rnn_hidden_dim]
        # (2) Enemy feature: [bs * n_enemies, enemy_fea_dim] -> [bs * n_enemies, enemy_feats_dim, rnn_hidden_dim * n_heads]
        input_w_enemy = self.hyper_input_w_goal(enemy_feats_t)
        input_b_enemy = self.hyper_input_b_goal(enemy_feats_t)
        # [bs * n_enemies, 1, enemy_fea_dim] * [bs * n_enemies, enemy_fea_dim, rnn_hidden_dim * head] = [bs * n_enemies, 1, rnn_hidden_dim * head]
        embedding_enemies = (th.matmul(enemy_feats_t.unsqueeze(1), input_w_enemy)+input_b_enemy).view(
            bs , self.goal_num, self.n_heads, self.hidden_sizes[0]
        )  # [bs, n_enemies, n_head, rnn_hidden_dim]
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs, n_head, rnn_hidden_dim]

        # (3) Ally features: [bs * n_allies, ally_fea_dim] -> [bs * n_allies, ally_feats_dim, rnn_hidden_dim * n_heads]
        input_w_ally = self.hyper_input_w_ally(ally_feats_t)
        # [bs * n_allies, 1, ally_fea_dim] * [bs * n_allies, ally_fea_dim, rnn_hidden_dim * head] = [bs * n_allies, 1, rnn_hidden_dim * head]
        embedding_allies = th.matmul(ally_feats_t.unsqueeze(1), input_w_ally).view(
            bs, self.ally_num, self.n_heads, self.hidden_sizes[0]
        )  # [bs, n_allies, n_head, rnn_hidden_dim]
        embedding_allies = embedding_allies.sum(dim=1, keepdim=False)  # [bs, n_head, rnn_hidden_dim]
        # Final embedding, merge multiple heads into one. -> [bs, n_head, rnn_hidden_dim]
        # (4) concatenate all features
        embedding = self.unify_input_heads(
            embedding_enemies + embedding_allies
        )
        # (5) MLP
        return self.fc2(embedding)


class MLPLayer1HPN(nn.Module):
    def __init__(self, input_dim, hidden_sizes, initialization_method, activation_func,hpn_structure):
        super(MLPLayer1HPN, self).__init__()

        active_func = get_active_func(activation_func)
        init_method = get_init_method(initialization_method)
        gain = nn.init.calculate_gain(activation_func)
        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)
        
        self.own_dim = hpn_structure[0]
        # self.goal_num = hpn_structure[1][0]#in lasercar, it's agent num, not used here
        self.goal_dim = hpn_structure[1][1]
        self.ally_num = hpn_structure[2][0]
        self.ally_dim = hpn_structure[2][1]
        self.n_heads = hpn_structure[3]
        self.hpn_hyper_dim = hpn_structure[4]
        self.hidden_sizes = hidden_sizes
        
        self.hyper_input_w_goal = Hypernet(
            input_dim=self.goal_dim, hidden_dim=self.hpn_hyper_dim,
            main_input_dim=self.goal_dim, main_output_dim=self.hidden_sizes[0],
            activation_func=activation_func, n_heads=self.n_heads
        )  # output shape: (enemy_feats_dim * self.hidden_sizes[0])
        self.hyper_input_b_goal = Hypernet(
            input_dim=self.goal_dim, hidden_dim=self.hpn_hyper_dim,
            main_input_dim=1, main_output_dim=self.hidden_sizes[0],
            activation_func=activation_func, n_heads=self.n_heads
        )  # output shape: (enemy_feats_dim * self.hidden_sizes[0])
        self.unify_input_heads = Merger(self.n_heads, self.hidden_sizes[0])

        self.fc2=nn.Sequential(*[
            get_active_func(activation_func),
            nn.LayerNorm(self.hidden_sizes[0]),
            init_(nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])),
            get_active_func(activation_func),
            nn.LayerNorm(self.hidden_sizes[1]),
        ])
    def forward(self, x):
        # (1) Own feature
        bs = x.size(0)
        self.goal_num = x.size(1) // self.goal_dim
        enemy_feats_t = x.reshape(-1,self.goal_dim)
        
        # embedding_own = self.fc1_own(own_feats_t)  # [bs , rnn_hidden_dim]
        # (2) Enemy feature: [bs * n_enemies, enemy_fea_dim] -> [bs * n_enemies, enemy_feats_dim, rnn_hidden_dim * n_heads]
        input_w_enemy = self.hyper_input_w_goal(enemy_feats_t)
        input_b_enemy = self.hyper_input_b_goal(enemy_feats_t)
        # [bs * n_enemies, 1, enemy_fea_dim] * [bs * n_enemies, enemy_fea_dim, rnn_hidden_dim * head] = [bs * n_enemies, 1, rnn_hidden_dim * head]
        embedding_enemies = (th.matmul(enemy_feats_t.unsqueeze(1), input_w_enemy)+input_b_enemy).view(
            bs , self.goal_num, self.n_heads, self.hidden_sizes[0]
        )  # [bs, n_enemies, n_head, rnn_hidden_dim]
        embedding_enemies = embedding_enemies.sum(dim=1, keepdim=False)  # [bs, n_head, rnn_hidden_dim]
        # (4) concatenate all features
        embedding = self.unify_input_heads(embedding_enemies)
        # (5) MLP
        return self.fc2(embedding)
    
class HPNMLPBase(nn.Module):
    """A MLP base module."""
    def __init__(self, args, obs_shape,hpn_structure):
        super(HPNMLPBase, self).__init__()
        # print(hpn_structure)
        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]
        self.hpn_structure=hpn_structure
        # hpn_structure [self.own_dim,[self.goal_num,self.goal_dim],[self.ally_num,self.ally_dim],self.n_heads,self.hpn_hyper_dim]
        obs_dim = obs_shape[0]
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
        assert len(hpn_structure) == 5, "hpn_structure should be a list of 5 elements"
        if hpn_structure[0] > 0:
            if hpn_structure[2][0] > 0:#[,[,],[,]]
                self.mlp = MLPLayerUni2HPN(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func,hpn_structure) 
                # if self.use_feature_normalization:
                #     self.feature_norm = nn.LayerNorm(obs_dim)
            else:#[,[0,0],[0,0]]
                self.mlp = MLPLayer(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func)
                # if self.use_feature_normalization:
                #     self.feature_norm = nn.LayerNorm(obs_dim)                
        else:#hpn_structure[0] = 0
            if hpn_structure[2][0] > 0:#[0,[,],[,]]
                self.mlp = MLPLayer2HPN(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func,hpn_structure)
                # if self.use_feature_normalization:
                #     self.feature_norm = nn.LayerNorm(obs_dim)                
            else:#[0,[,],[0,0]]
                # print(hpn_structure)
                self.mlp = MLPLayer1HPN(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func,hpn_structure)
                # if self.use_feature_normalization:
                #     self.feature_norm = nn.LayerNorm(hpn_structure[1][1])                
        # print(self.mlp,"parameter num",[p.numel() for p in self.mlp.parameters()],np.sum([p.numel() for p in self.mlp.parameters()]))
    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)
            # if self.hpn_structure[0] > 0:
            #     if self.hpn_structure[2][0] > 0:#[,[,],[,]]
            #         x = self.feature_norm(x)
            #     else:#[,[0,0],[0,0]]
            #         x = self.feature_norm(x)              
            # else:#hpn_structure[0] = 0
            #     if self.hpn_structure[2][0] > 0:#[0,[,],[,]]
            #         x = self.feature_norm(x)               
            #     else:#[0,[,],[0,0]]
            #         # print(x.shape,x.reshape(-1,self.hpn_structure[1][1]).shape)
            #         x = self.feature_norm(x.reshape(-1,self.hpn_structure[1][1])).view(-1,self.hpn_structure[1][0]*self.hpn_structure[1][1])
            #         # print(x.shape)
        x = self.mlp(x)
        return x
import numpy as np
class DeepSetMLPBase(nn.Module):
    """A MLP base module."""
    def __init__(self, args, obs_shape,DeepSet_structure):
        super(DeepSetMLPBase, self).__init__()
        self.use_feature_normalization = args["use_feature_normalization"]
        self.initialization_method = args["initialization_method"]
        self.activation_func = args["activation_func"]
        self.hidden_sizes = args["hidden_sizes"]
        # DeepSet_structure [self.own_dim,[self.goal_num,self.goal_dim],[self.ally_num,self.ally_dim]]
        obs_dim = obs_shape[0]
        if self.use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)
        # print(DeepSet_structure)
        assert len(DeepSet_structure) == 3, "DeepSet_structure should be a list of 3 elements"
        if DeepSet_structure[0] > 0:
            if DeepSet_structure[2][0] > 0:#[,[,],[,]]
                self.mlp = MLPLayerUni2DeepSet(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func,DeepSet_structure) 
            elif DeepSet_structure[1][0] > 0:#[,[,],[0,0]]
                self.mlp = MLPLayerUni1DeepSet(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func,DeepSet_structure)
            else:#[,[0,0],[0,0]]
                self.mlp = MLPLayer(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func)
        else:#DeepSet_structure[0] = 0
            if DeepSet_structure[2][0] > 0:#[0,[,],[,]]
                self.mlp = MLPLayer2DeepSet(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func,DeepSet_structure)
            else:#[0,[,],[0,0]]
                self.mlp = MLPLayer1DeepSet(obs_dim, self.hidden_sizes, self.initialization_method, self.activation_func,DeepSet_structure)
        # print(self.mlp,"parameter num",[p.numel() for p in self.mlp.parameters()],np.sum([p.numel() for p in self.mlp.parameters()]))
    def forward(self, x):
        if self.use_feature_normalization:
            x = self.feature_norm(x)
        x = self.mlp(x)
        return x

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # self.attn = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, n_block, n_embd, n_head, n_agent ):
        super(Encoder, self).__init__()

        self.state_dim = state_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        # self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
        #                           init_(nn.Linear(n_embd, 1)))

    def forward(self, state):
        # state: (batch, n_agent, state_dim)
        # if len(state.shape)==2:
        state = state.unsqueeze(0)
        # else: print("state shape",state.shape)
        state_embeddings = self.state_encoder(state)
        rep = self.blocks(self.ln(state_embeddings)).squeeze(0)
        # v_loc = self.head(rep)
        # print(rep.shape,state.shape,state_embeddings.shape)
        return  rep
    
    
if __name__ == "__main__":

    # Encoder=Encoder(state_dim=62, n_block=2, n_embd=64, n_head=4, n_agent=10)
    # obs = torch.randn(21, 62)
    # rep = Encoder(obs)
    # print(Encoder,"parameter num",[p.numel() for p in Encoder.parameters()],np.sum([p.numel() for p in Encoder.parameters()]))
      # Should be [10, 1]
    
    # obs_shape=(62,)
    # obs = torch.randn(10, 62)
    # hpn_structure = [35, [3, 5], [4, 3], 1, 64]
    
    # obs_shape=(62,)
    # obs = torch.randn(10, 62)
    # hpn_structure = [62, [0, 5], [0, 3], 1, 128]  

    # obs_shape=(27,)
    # obs = torch.randn(10, 27)
    # hpn_structure = [0, [3, 5], [4, 3], 1, 64]      
    
    # obs_shape=(30,)
    # obs = torch.randn(10, 30)
    # hpn_structure = [15, [3, 5], [0, 3], 1, 64]

    obs_shape=(15,)
    obs = torch.randn(10, 15)
    hpn_structure = [0, [3, 5], [0, 3], 1, 64]
     
    model = HPNMLPBase({ 'hidden_sizes': [128, 128], 'activation_func': 'relu', 'use_feature_normalization': True, 'initialization_method': 'orthogonal_', 'gain': 0.01}, obs_shape,hpn_structure=hpn_structure,)
    output = model(obs)
    print(output.shape)  # Should be [10, 20]
    
    
    # obs_shape=(62,)
    # obs = torch.randn(10, 62)
    # hpn_structure = [35, [3, 5], [4, 3]]
    
    # obs_shape=(62,)
    # obs = torch.randn(10, 62)
    # hpn_structure = [62, [0, 5], [0, 3], 1, 128]  

    # obs_shape=(27,)
    # obs = torch.randn(10, 27)
    # hpn_structure = [0, [3, 5], [4, 3]]  
    # obs_shape=(27,)
    # obs = torch.randn(10, 27)
    # hpn_structure = [27, [0, 5], [0, 3], 1, 64]       
    
    # obs_shape=(15,)
    # obs = torch.randn(10, 15)
    # hpn_structure = [0, [3, 5], [0, 3]]
    # obs_shape=(15,)
    # obs = torch.randn(10, 15)
    # hpn_structure = [15, [0, 5], [0, 3]]
    # obs_shape=(18,)
    # obs = torch.randn(10, 18)
    # hpn_structure = [15, [1, 3], [0, 3]]     
    # model = DeepSetMLPBase({ 'hidden_sizes': [128, 128], 'activation_func': 'relu', 'use_feature_normalization': True, 'initialization_method': 'orthogonal_', 'gain': 0.01}, obs_shape,DeepSet_structure=hpn_structure,)
    # output = model(obs)
    # print(output.shape)  # Should be [10, 20]


