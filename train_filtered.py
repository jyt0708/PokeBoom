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

from torch.utils.data import Dataset
import numpy as np
import ast
import glob

class TxtReplayDataset(Dataset):
    def __init__(self, folder_path, observation_space, action_space, reward_function):
        self.files = sorted(glob.glob(os.path.join(folder_path, "round_*.txt")))
        self.obs_space = observation_space
        self.action_space = action_space
        self.reward_fn = reward_function

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        with open(file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()

        # Parse the dict safely
        data = ast.literal_eval(raw_content)

        # Extract elements
        obs_seq = {
            "text": data["Observation sequence"]["text"],
            "numbers": np.array(data["Observation sequence"]["numbers"])
        }
        action_seq = {
            "chosen": data["Action"]["chosen"],
            "missing": data["Action"]["missing"],
            "llm_output": data["Action"]["Output"]
        }
        reward_seq = data["Reward"]
        done_seq = data["Done"]

        return obs_seq, action_seq, reward_seq, done_seq


# Replace ParsedReplayDataset with your custom dataset
offline_dset = TxtReplayDataset(
    folder_path="offline_with_llm_outputs",   # path to folder with round_xxx.txt
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
)

# Use only the subset for faster testing/training
offline_subset = Subset(offline_dset, range(1))

# Custom collate function for variable-length sequences
def custom_collate(batch):
    obs_seqs, action_seqs, reward_seqs, done_seqs = zip(*batch)
    return list(obs_seqs), list(action_seqs), list(reward_seqs), list(done_seqs)

def save_checkpoint(model, optimizer, epoch, batch_idx, avg_weighted_loss=None):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    checkpoint_data = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }

    if avg_weighted_loss is not None:
        checkpoint_data["avg_weighted_loss"] = avg_weighted_loss

    torch.save(checkpoint_data, CHECKPOINT_PATH)

def load_checkpoint(model, optimizer):
    if os.path.exists(CHECKPOINT_PATH):
        checkpoint = torch.load(CHECKPOINT_PATH)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        print(f"🔄 Resumed from epoch {checkpoint['epoch']}, batch {checkpoint['batch_idx']}")
        return checkpoint["epoch"], checkpoint["batch_idx"]
    return 0, 0

policy = HybridPolicy(
    num_dim=obs_space.gym_space["numbers"].shape[0],
    action_dim=action_space.gym_space.n
)
optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
# Loss with ignore_index for missing actions
loss_fn = nn.CrossEntropyLoss(reduction="none", ignore_index=-1)

CHECKPOINT_PATH = "checkpoint/latest.pt"

# Training loop
# for epoch in range(10):
start_epoch, start_batch = load_checkpoint(policy, optimizer)

for epoch in range(start_epoch, 6):
    running_weighted_loss = 0.0
    running_ce_loss = 0.0
    batch_count = 0

    for batch_idx, (obs_seqs, action_seqs, reward_seqs, _) in enumerate(
        DataLoader(offline_dset, batch_size=64, shuffle=True, collate_fn=custom_collate)
    ):
        if epoch == start_epoch and batch_idx < start_batch:
            continue

        obs_seq = obs_seqs[0]
        action_seq = action_seqs[0]

        llm_batch = action_seq["llm_output"]
        num_batch = torch.tensor(obs_seq["numbers"], dtype=torch.float32)
        actions = torch.tensor(action_seq['chosen'], dtype=torch.long)

        mask = torch.tensor([not m for m in action_seq["missing"]], dtype=torch.bool)
        if mask.sum() == 0:
            continue

        num_batch_masked = num_batch[:-1][mask]
        llm_batch_masked = [llm for llm, m in zip(llm_batch[:-1], mask) if m]
        actions_masked = actions[mask]

        try:
            logits = policy(llm_batch_masked, num_batch_masked)

            ce_loss = loss_fn(logits, actions_masked)
            weighted_loss = (ce_loss * 1.0).mean()  # 去掉 reward 权重
            avg_ce_loss = ce_loss.mean().item()

            optimizer.zero_grad()
            weighted_loss.backward()
            optimizer.step()

            running_weighted_loss += weighted_loss.item()
            running_ce_loss += avg_ce_loss
            batch_count += 1

            if batch_idx % 50 == 0 and batch_count > 0:
                avg_weighted_loss = running_weighted_loss / batch_count
                avg_ce_loss_batch = running_ce_loss / batch_count
                save_checkpoint(policy, optimizer, epoch, batch_idx,
                                avg_weighted_loss=avg_weighted_loss)
                print(
                    f"Epoch {epoch}, Batch {batch_idx}: "
                    f"Avg Weighted Loss={avg_weighted_loss:.4f}, "
                    f"Avg CE Loss={avg_ce_loss_batch:.4f}"
                )

        except Exception as e:
            print(f"⚠️ Error at epoch {epoch}, batch {batch_idx}: {e}")
            save_checkpoint(policy, optimizer, epoch, batch_idx, weighted_loss.item())
            raise

    if batch_count > 0:
        print(
            f"Epoch {epoch} completed: Avg Weighted Loss={running_weighted_loss/batch_count:.4f}, "
            f"Avg CE Loss={running_ce_loss/batch_count:.4f}"
        )

# Save the trained model
torch.save(policy.state_dict(), "test_hybrid_policy.pt")
print("Model saved to hybrid_policy.pt")
