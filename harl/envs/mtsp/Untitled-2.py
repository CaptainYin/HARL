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
        self.city_encoder = GatedTransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)
        # attention-based aggregation blocks
        self.agent_agent_encoder = GatedTransformerEncoder(2, d_model, nhead, dim_feedforward, dropout)
        self.agent_city_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, cities, agent_idx_mask):
        """
        cities: [batch, num_cities, 2]
        agent_idx_mask: [batch, num_agents] -> current city idx per agent
        """
        batch, N, _ = cities.size()
        # embed cities
        e = self.city_proj(cities)                      # [batch, N, d_model]
        e = e.transpose(0,1)                           # [N, batch, d_model]
        e_enc = self.city_encoder(e)                   # [N, batch, d_model]

        # global graph info
        g = e_enc.mean(dim=0, keepdim=True)            # [1, batch, d_model]

        # build agent state features
        depot = e_enc[0].unsqueeze(0)                  # [1, batch, d_model]
        curr = []
        for idxs in agent_idx_mask.transpose(0,1):
            curr.append(e_enc[idxs].unsqueeze(0))      # [1, batch, d_model]
        curr = torch.cat(curr, dim=0)                  # [num_agents, batch, d_model]
        # combine
        s = depot + curr + g                           # [num_agents, batch, d_model]

        # agent-to-agent aggregation
        s = self.agent_agent_encoder(s)
        # agent-to-city cross-attn
        q = s                                         # [M, B, D]
        k = v = e_enc                                 # [N, B, D]
        attn_out, _ = self.agent_city_attn(q, k, v)
        s = s + attn_out                              # gated skip could be added
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
        # agent_feats: [M, B, D], city_feats: [N, B, D], visited_mask: [B, N]
        M, B, D = agent_feats.size()
        N, _, _ = city_feats.size()
        Q = self.q_proj(agent_feats)                  # [M, B, D]
        K = self.k_proj(city_feats)                   # [N, B, D]
        # compat [B, N, M]
        logits = torch.einsum('mbd,nbd->bnm', Q, K) / (D**0.5)
        # mask visited cities
        logits = logits.masked_fill(~visited_mask.unsqueeze(-1), float('-inf'))
        logits = logits.view(B, -1)                   # [B, N*M]
        logits = self.C * torch.tanh(logits)
        probs = F.softmax(logits, dim=-1)
        return probs.view(B, M, N)

# ---------------------------
# Environment Interface
# ---------------------------
class MTSPEnvironment:
    def __init__(self, num_cities, num_agents):
        self.num_cities = num_cities
        self.num_agents = num_agents

    def sample(self, batch_size):
        # Random city coords in [0,1]^2, depot is index 0
        self.city_coords = torch.rand(batch_size, self.num_cities, 2)
        # all agents start at depot (idx 0)
        self.agent_idx = torch.zeros(batch_size, self.num_agents, dtype=torch.long)
        # visited mask: depot is considered visited
        self.visited = torch.zeros(batch_size, self.num_cities, dtype=torch.bool)
        self.visited[:,0] = True
        return self.city_coords, self.agent_idx.clone()

    def step(self, agent_choice, city_choice):
        # agent_choice, city_choice: [batch]
        B = agent_choice.size(0)
        batch_idx = torch.arange(B)
        # old positions
        old_idx = self.agent_idx[batch_idx, agent_choice]
        old_pos = self.city_coords[batch_idx, old_idx]
        # new positions
        new_pos = self.city_coords[batch_idx, city_choice]
        # compute distance reward
        step_rewards = torch.norm(old_pos - new_pos, dim=1)
        # update state
        self.agent_idx[batch_idx, agent_choice] = city_choice
        self.visited[batch_idx, city_choice] = True
        # return positions [B, M, 2], per-batch reward, done flag
        agent_positions = self.city_coords[torch.arange(B).unsqueeze(-1), self.agent_idx]
        done = self.visited.all()
        return agent_positions, step_rewards, done

# ---------------------------
# AMARL Model & Training Skeleton
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

    def forward(self, city_coords, agent_idx, visited_mask):
        # city_coords: [B, N, 2], agent_idx: [B, M], visited_mask: [B, N]
        s_feats, c_feats = self.sf_extractor(city_coords, agent_idx)
        probs = self.selector(s_feats, c_feats, ~visited_mask)
        return probs


def train_amarl(env, model, optimizer, epochs, batch_size):
    for ep in range(epochs):
        city_coords, agent_idx = env.sample(batch_size)
        visited = env.visited.clone()
        log_probs, rewards = [], []
        done = False; step = 0
        while not done and step < model.num_cities:
            probs = model(city_coords, agent_idx, visited)
            B, M, N = probs.size()
            dist = Categorical(probs.view(B, -1))
            action = dist.sample()
            logp = dist.log_prob(action)
            log_probs.append(logp)
            ag = action // N; ct = action % N
            agent_idx = ag.unsqueeze(-1)
            agent_positions, step_r, done = env.step(ag, ct)
            rewards.append(step_r)
            visited = env.visited.clone()
            step += 1
        total_reward = torch.stack(rewards).sum(dim=0)
        # use max for min-max objective
        team_return = total_reward.max()
        loss = (torch.stack(log_probs).sum(dim=0) * team_return).mean()
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        print(f"Epoch {ep}: Loss={loss.item():.4f}, Return={team_return.item():.4f}")

# Example usage:
# env = MTSPEnvironment(num_cities=50, num_agents=5)
# model = AMARL(50,5)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
# train_amarl(env, model, optimizer, epochs=100, batch_size=32)
