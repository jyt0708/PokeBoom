import sys
import os
# Ensure the inner metamon/ (the actual package) is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "metamon"))

from HybridPolicy import HybridPolicy
from metamon.env import get_metamon_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace
from metamon.data import ParsedReplayDataset

from torch.utils.data import Subset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

os.environ["METAMON_CACHE_DIR"] = "metamon_cache" # "C:/Users/jytna7OneDrive - TUM/HiWi/PokeBoom/metamon_cache"

# init metamon components
team_set = get_metamon_teams("gen1ou", "competitive")
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()

# pytorch dataset. examples are converted to 
# the chosen obs/actions/rewards on-the-fly.
offline_dset = ParsedReplayDataset(
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
    formats=["gen1ou"],
)

# Use only the subset for faster testing/training
offline_subset = Subset(offline_dset, range(1))

# Custom collate function for variable-length sequences
def custom_collate(batch):
    obs_seqs, action_seqs, reward_seqs, done_seqs = zip(*batch)
    return list(obs_seqs), list(action_seqs), list(reward_seqs), list(done_seqs)
    
policy = HybridPolicy(
    "distilbert-base-uncased",
    num_dim=obs_space.gym_space["numbers"].shape[0],
    action_dim=action_space.gym_space.n
)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
# Loss with ignore_index for missing actions
loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

# Training loop
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
torch.save(policy.state_dict(), "test_hybrid_policy.pt")
print("Model saved to hybrid_policy.pt")