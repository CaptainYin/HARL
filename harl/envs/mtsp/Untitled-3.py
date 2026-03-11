"""
Implementation of AMARL: Attention-Based Multiagent RL for Min-Max MTSP
Based on: Gao et al., IEEE TNNLS, 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# ---------------------------
# Gated Transformer Encoder
# ---------------------------
class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.gate1 = nn.Linear(d_model * 2, d_model)
        self.gate2 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def gated(self, x, y, gate):
        g = torch.sigmoid(gate(torch.cat([x, y], dim=-1)))
        return x + g * y

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        q = k = self.norm1(src)
        attn_output, _ = self.self_attn(q, k, src,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        src2 = self.gated(src, self.dropout(attn_output), self.gate1)
        src_ff = self.norm2(src2)
        ff = self.linear2(F.relu(self.linear1(src_ff)))
        out = self.gated(src2, self.dropout(ff), self.gate2)
        return out

class GatedTransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            GatedTransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None, padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=padding_mask)
        return x

# ---------------------------
# State Feature Extraction
# ---------------------------
class StateFeatureExtractor(nn.Module):
    def __init__(self, city_dim, d_model, num_layers, nhead, dim_feedforward, dropout):
        super().__init__()
        self.city_proj = nn.Linear(2, d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.city_encoder = GatedTransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        self.agent_agent_encoder = GatedTransformerEncoder(2, d_model, nhead, dim_feedforward, dropout)
        self.agent_city_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, cities, agent_pos_mask):
        batch, N, _ = cities.size()
        e = self.city_proj(cities)
        e = e.transpose(0,1)
        e_enc = self.city_encoder(e)
        g = e_enc.mean(dim=0, keepdim=True)
        depot = e_enc[0].unsqueeze(0)
        curr = []
        for i, idxs in enumerate(agent_pos_mask.transpose(0,1)):
            curr.append(e_enc[idxs].unsqueeze(0))
        curr = torch.cat(curr, dim=0)
        s = depot + curr + g
        s = self.agent_agent_encoder(s)
        q = s
        k = v = e_enc
        attn_out, _ = self.agent_city_attn(q, k, v)
        s = s + attn_out
        return s, e_enc

# ---------------------------
# Action Selection Network
# ---------------------------
class ActionSelector(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.C = 10

    def forward(self, agent_feats, city_feats, visited_mask):
        M, batch, d = agent_feats.size()
        N, _, _ = city_feats.size()
        Q = self.q_proj(agent_feats)
        K = self.k_proj(city_feats)
        logits = torch.einsum('mbd,nbd->bnm', Q, K) / (d**0.5)
        logits = logits.masked_fill(~visited_mask.unsqueeze(-1), float('-inf'))
        logits = logits.view(batch, -1)
        logits = self.C * torch.tanh(logits)
        probs = F.softmax(logits, dim=-1)
        return probs.view(batch, M, N)

# ---------------------------
# AMARL Agent / Training
# ---------------------------
class AMARL(nn.Module):
    def __init__(self, num_cities, num_agents, d_model=128, nhead=8,
                 city_enc_layers=3, agg_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.sf_extractor = StateFeatureExtractor(num_cities, d_model,
                                                  city_enc_layers, nhead,
                                                  dim_feedforward, dropout)
        self.selector = ActionSelector(d_model)
        self.num_agents = num_agents
        self.num_cities = num_cities

    def forward(self, city_coords, agent_mask, visited_mask):
        s_feats, c_feats = self.sf_extractor(city_coords, agent_mask)
        probs = self.selector(s_feats, c_feats, visited_mask)
        return probs

# ---------------------------
# MTSP Environment Interface
# ---------------------------
class MTSPEnv:
    def __init__(self, num_agents, num_cities):
        self.M = num_agents
        self.N = num_cities
        self.depot_idx = 0

    def sample(self, batch_size):
        coords = torch.rand(batch_size, self.N, 2)
        agent_pos = torch.zeros(batch_size, self.M, dtype=torch.long)  # all start at depot
        return coords, agent_pos

    def step(self, agent_idx, city_idx):
        """
        Simulate transition:
        agent_idx, city_idx: [batch] tensor
        returns:
          - next positions of agents (unused placeholder for now)
          - per-agent step distance (reward)
          - done flag (if all cities visited)
        """
        # Placeholder logic: distance from agent current pos to target city
        # Replace with actual environment state tracking for real applications
        B = agent_idx.shape[0]
        step_rewards = torch.rand(B)  # mock reward
        done = torch.tensor([False]*B)  # set to True when all cities visited
        return None, step_rewards, done

# Training loop skeleton

def train_amarl(env, model, optimizer, epochs, batch_size):
    for ep in range(epochs):
        city_coords, init_agent_mask = env.sample(batch_size)
        visited = torch.zeros(batch_size, model.num_cities, dtype=torch.bool)
        agent_mask = init_agent_mask.clone()
        log_probs = []
        rewards = []

        done = False
        step = 0
        while not done and step < model.num_cities:
            probs = model(city_coords, agent_mask, ~visited)
            m, n = model.num_agents, model.num_cities
            probs_flat = probs.view(batch_size, -1)
            dist = Categorical(probs_flat)
            action = dist.sample()
            logp = dist.log_prob(action)
            log_probs.append(logp)
            agent_idx = action // n
            city_idx = action % n
            agent_mask = agent_idx.unsqueeze(-1)
            visited[torch.arange(batch_size), city_idx] = True
            _, step_rewards, done = env.step(agent_idx, city_idx)
            rewards.append(step_rewards)
            step += 1

        total_reward = torch.stack(rewards, dim=0).sum(dim=0).max()
        loss = (torch.stack(log_probs, dim=0).sum(dim=0) * total_reward).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {ep}: Loss {loss.item():.4f}, Return {total_reward.item():.4f}")
