"""
Implementation of AMARL: Attention-Based Multiagent RL for Min-Max MTSP
Based on: Gao et al., IEEE TNNLS, 2024
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# ---------------------------
# Gated Transformer Encoder
# ---------------------------
class GatedTransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        # Gating
        self.gate1 = nn.Linear(d_model * 2, d_model)
        self.gate2 = nn.Linear(d_model * 2, d_model)
        # LayerNorm inside
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def gated(self, x, y, gate):
        # x, y: [seq, batch, d_model]
        g = torch.sigmoid(gate(torch.cat([x, y], dim=-1)))
        return x + g * y

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # src: [seq_len, batch, d_model]
        # Self-attention
        q = k = self.norm1(src)
        attn_output, _ = self.self_attn(q, k, src,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        src2 = self.gated(src, self.dropout(attn_output), self.gate1)
        # Feedforward
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
        # x: [seq_len, batch, d_model]
        for layer in self.layers:
            x = layer(x, src_mask=mask, src_key_padding_mask=padding_mask)
        return x

# ---------------------------
# State Feature Extraction
# ---------------------------
class StateFeatureExtractor(nn.Module):
    def __init__(self, city_dim, d_model, num_layers, nhead, dim_feedforward, dropout):
        super().__init__()
        # project 2D coords to d_model
        self.city_proj = nn.Linear(2, d_model)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.city_encoder = GatedTransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        # attention-based aggregation blocks
        self.agent_agent_encoder = GatedTransformerEncoder(2, d_model, nhead, dim_feedforward, dropout)
        self.agent_city_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, cities, agent_pos_mask):
        """
        cities: [batch, num_cities, 2]
        agent_pos_mask: [batch, num_agents] -> indices of current cities for each agent
        """
        batch, N, _ = cities.size()
        # embed cities
        e = self.city_proj(cities)                      # [batch, N, d_model]
        e = e.transpose(0,1)                           # [seq=N, batch, d_model]
        e_enc = self.city_encoder(e)                   # [N, batch, d_model]

        # global graph info
        g = e_enc.mean(dim=0, keepdim=True)            # [1, batch, d_model]

        # build agent state features
        # gather embedding of depot and current pos
        # assume depot idx=0
        depot = e_enc[0].unsqueeze(0)                  # [1, batch, d_model]
        curr = []
        for i, idxs in enumerate(agent_pos_mask.transpose(0,1)):
            curr.append(e_enc[idxs].unsqueeze(0))      # list of [1, batch, d_model]
        curr = torch.cat(curr, dim=0)                  # [num_agents, batch, d_model]
        # concat depot+curr + global
        s = depot + curr + g                           # [num_agents, batch, d_model]

        # agent-to-agent aggregation
        s = self.agent_agent_encoder(s)
        # agent-to-city cross-attn
        q = s                                        # [num_agents, batch, d_model]
        k = v = e_enc                                # [N, batch, d_model]
        attn_out, _ = self.agent_city_attn(q, k, v)
        # gated combine
        s = s + attn_out                             # [num_agents, batch, d_model]
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
        # agent_feats: [num_agents, batch, d_model]
        # city_feats: [num_cities, batch, d_model]
        M, batch, d = agent_feats.size()
        N, _, _ = city_feats.size()
        # compute attention logits
        Q = self.q_proj(agent_feats)  # [M, B, D]
        K = self.k_proj(city_feats)   # [N, B, D]
        # compute compatibility
        logits = torch.einsum('mbd,nbd->bnm', Q, K) / (d**0.5)  # [batch, N, M]
        # apply visited mask
        logits = logits.masked_fill(~visited_mask.unsqueeze(-1), float('-inf'))
        # flatten
        logits = logits.view(batch, -1)                     # [batch, N*M]
        # clip
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
        # city_coords: [batch, N, 2]
        # agent_mask: [batch, M] idx of current city per agent
        # visited_mask: [batch, N] boolean
        s_feats, c_feats = self.sf_extractor(city_coords, agent_mask)
        probs = self.selector(s_feats, c_feats, visited_mask)
        return probs

# Training loop skeleton

def train_amarl(env, model, optimizer, epochs, batch_size):
    for ep in range(epochs):
        # sample batch of instances
        city_coords, init_agent_mask = env.sample(batch_size)
        visited = torch.zeros(batch_size, model.num_cities, dtype=torch.bool)
        agent_mask = init_agent_mask.clone()
        log_probs = []
        rewards = []

        done = False
        step = 0
        while not done and step < model.num_cities:
            probs = model(city_coords, agent_mask, ~visited)
            # sample joint action: flatten MxN
            m, n = model.num_agents, model.num_cities
            probs_flat = probs.view(batch_size, -1)
            dist = Categorical(probs_flat)
            action = dist.sample()
            logp = dist.log_prob(action)
            log_probs.append(logp)
            # decode action to (agent, city)
            agent_idx = action // n
            city_idx = action % n
            # update agent positions and visited
            agent_mask = agent_idx.unsqueeze(-1)
            visited[torch.arange(batch_size), city_idx] = True
            # compute step reward
            # assume env.step returns next agent coords and reward
            agent_positions, step_rewards, done = env.step(agent_idx, city_idx)
            rewards.append(step_rewards)
            step += 1

        # compute final reward: max path length
        total_reward = torch.stack(rewards, dim=0).sum(dim=0).max()
        loss = (torch.stack(log_probs, dim=0).sum(dim=0) * total_reward).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {ep}: Loss {loss.item():.4f}, Return {total_reward.item():.4f}")
