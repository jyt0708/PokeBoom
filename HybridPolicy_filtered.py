import torch
import torch.nn as nn
import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

class HybridPolicy(nn.Module):
    def __init__(self, num_dim, action_dim=8, hidden_dim=256, action_emb_dim=32):
        super().__init__()
        self.num_dim = num_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_emb_dim = action_emb_dim

        # Numeric encoder
        self.num_encoder = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        # Trainable embeddings for LLM-suggested actions
        self.action_embedding = nn.Embedding(action_dim, action_emb_dim)

        # Fusion head
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim + action_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Fixed action mapping
        self.action_map = {
            "switch to slot 1": 0,
            "switch to slot 2": 1,
            "switch to slot 3": 2,
            "switch to slot 4": 3,
            "use move 1": 4,
            "use move 2": 5,
            "use move 3": 6,
            "use move 4": 7,
        }

    def forward(self, llm_batch, num_batch):
        """
        text_batch: list of battle states (strings)
        num_batch: tensor [batch_size, num_dim]
        """
        device = num_batch.device

        # Convert LLM outputs → action ids
        action_ids = [self.action_map.get(txt, 0) for txt in llm_batch]
        action_ids = torch.tensor(action_ids, dtype=torch.long, device=device)

        # Encode numbers
        num_emb = self.num_encoder(num_batch)

        # Encode action embeddings
        action_emb = self.action_embedding(action_ids)

        # Fuse numeric + LLM signals
        fused = torch.cat([num_emb, action_emb], dim=-1)
        logits = self.fusion(fused)

        return logits