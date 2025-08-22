import sys
import os

# Ensure the inner metamon/ (the actual package) is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "metamon"))

from metamon.env import get_metamon_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace

team_set = get_metamon_teams("gen1ou", "competitive")
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()

from metamon.data import ParsedReplayDataset
# pytorch dataset. examples are converted to 
# the chosen obs/actions/rewards on-the-fly.
offline_dset = ParsedReplayDataset(
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
    formats=["gen1ou"],
)

# Use only the first 50 samples for faster testing/training
from torch.utils.data import Subset
offline_subset = Subset(offline_dset, range(50))

# If you need num_obs for the subset only:
# num_obs = num_obs = [obs for sample in offline_dset for obs in sample[0]["numbers"]]
num_obs = [obs for sample in offline_subset for obs in sample[0]["numbers"]]
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
# Custom collate function for variable-length sequences
def custom_collate(batch):
    obs_seqs, action_seqs, reward_seqs, done_seqs = zip(*batch)
    return list(obs_seqs), list(action_seqs), list(reward_seqs), list(done_seqs)
from transformers import AutoTokenizer, AutoModel

class HybridPolicy(nn.Module):
    def __init__(self, llm_name, num_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Load pretrained LLM (encoder only)
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm = AutoModel.from_pretrained(llm_name)  # e.g., "distilbert-base-uncased"
        
        llm_out_dim = self.llm.config.hidden_size
        
        # Numeric MLP encoder
        self.num_encoder = nn.Sequential(
            nn.Linear(num_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Fusion + action head
        self.fusion = nn.Sequential(
            nn.Linear(llm_out_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
    
    def forward(self, text_batch, num_batch):
        # Encode text via LLM (CLS embedding or mean pooling)
        inputs = self.tokenizer(text_batch, return_tensors="pt", padding=True, truncation=True)
        outputs = self.llm(**inputs)
        text_emb = outputs.last_hidden_state.mean(dim=1)  # [B, hidden]
        
        # Encode numbers
        num_emb = self.num_encoder(num_batch)
        
        # Fuse
        fused = torch.cat([text_emb, num_emb], dim=-1)
        logits = self.fusion(fused)
        return logits
    
policy = HybridPolicy(
    "distilbert-base-uncased",
    num_dim=obs_space.gym_space["numbers"].shape[0],
    action_dim=action_space.gym_space.n
)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
# Loss with ignore_index for missing actions
loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

# for epoch in range(10):
for epoch in range(1):
    total_loss = 0
    for obs_seqs, action_seqs, reward_seqs, _ in DataLoader(offline_subset, batch_size=1, shuffle=True, collate_fn=custom_collate):
        # Since batch_size=1, each is a list of length 1
        obs_seq = obs_seqs[0]
        action_seq = action_seqs[0]
        reward_seq = reward_seqs[0]
        # Convert each component of text_batch to a single string
        text_batch = [str(x) for x in obs_seq["text"]]
        import numpy as np
        num_batch = torch.tensor(obs_seq["numbers"], dtype=torch.float32)
        # Ensure num_batch is [T, n_features] to match action_seq length
        actions = torch.tensor(action_seq['chosen'], dtype=torch.long)
        rewards = torch.tensor(reward_seq, dtype=torch.float32)
        print(f"text_batch shape: {len(text_batch)}, num_batch shape: {num_batch.shape}, action_seq length: {len(actions)}, reward_seq length: {len(rewards)}")

        # Mask for valid timesteps
        mask = torch.tensor([not m for m in action_seq["missing"]], dtype=torch.bool)
        if mask.sum() == 0:
            continue  # skip if all actions are missing

        # Filter valid timesteps
        text_batch_masked = [text for text, m in zip(text_batch[:-1], mask) if m]
        num_batch_masked = num_batch[:-1][mask]
        actions_masked = actions[mask]
        rewards_masked = rewards[mask]

        print(f"The number of valid samples: {mask.sum()}")
        print(f"text_batch_masked length: {len(text_batch_masked)}")
        print(f"num_batch_masked shape: {num_batch_masked.shape}")
        print(f"actions_masked length: {len(actions_masked)}")
        print(f"rewards_masked length: {len(rewards_masked)}")

        # Forward pass
        logits = policy(text_batch_masked, num_batch_masked)

        # Compute reward-weighted BC loss
        ce_loss = loss_fn(logits, actions_masked)
        weighted_loss = (ce_loss * (1.0 + rewards_masked)).mean()
        
        optimizer.zero_grad()
        weighted_loss.backward()
        optimizer.step()
        
        total_loss += weighted_loss.item()
    
    print(f"Epoch {epoch}: Loss={total_loss:.4f}")

# Save the trained model
torch.save(policy.state_dict(), "hybrid_policy.pt")
print("Model saved to hybrid_policy.pt")

def select_action(obs):
    text = obs["text"]
    numbers = torch.tensor(obs["numbers"], dtype=torch.float32).unsqueeze(0)
    
    logits = policy([text], numbers)
    action = torch.argmax(logits, dim=-1).item()
    return action