import sys
import os
# Ensure the inner metamon/ (the actual package) is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "metamon"))

from HybridPolicy import HybridPolicy
from metamon.env import get_metamon_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace
from metamon.env import PokeAgentLadder
from metamon.data import ParsedReplayDataset

import torch
from dotenv import load_dotenv
load_dotenv()

# $env:METAMON_CACHE_DIR = "metamon_cache"
os.environ["METAMON_CACHE_DIR"] = "metamon_cache"

team_set = get_metamon_teams("gen1ou", "competitive")
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()

env = PokeAgentLadder(
    battle_format="gen1ou",
    player_username=os.getenv("PLAYER_USERNAME"),
    player_password=os.getenv("PLAYER_PASSWORD"),
    num_battles=2,
    observation_space=obs_space,
    action_space=action_space,
    reward_function=reward_fn,
    player_team_set=team_set,
    save_trajectories_to=None,
)

# online_dset = ParsedReplayDataset(
#     dset_root="online_data",
#     observation_space=obs_space,
#     action_space=action_space,
#     reward_function=reward_fn,
# )

policy = HybridPolicy(
    "distilbert-base-uncased",
    num_dim=obs_space.gym_space["numbers"].shape[0],
    action_dim=action_space.gym_space.n
)
policy.load_state_dict(torch.load("hybrid_policy.pt"))
policy.eval()

def select_action(obs):
    if isinstance(obs, tuple):
        obs_dict, info = obs
    else:
        obs_dict = obs
        # fallback info if not provided
        info = {"legal_actions": list(range(action_space.gym_space.n))}

    # Convert the long string into a batch of size 1
    text_str = str(obs_dict["text"])       
    text = [text_str]
    
    numbers = torch.tensor(obs_dict["numbers"], dtype=torch.float32).unsqueeze(0)
    
    logits = policy(text, numbers)
    
    # Mask illegal actions
    # legal = info["legal_actions"]
    # legal_logits = logits[0, legal]
    # best_idx = torch.argmax(legal_logits).item()
    # action = legal[best_idx]
    action = torch.argmax(logits, dim=1).item()

    return action

for battle_idx in range(env.num_battles):
    print(f"Battle {battle_idx+1} starting...")
    obs = env.reset()
    terminated = False
    while not terminated:
        action = select_action(obs)
        obs, reward, terminated, _, _ = env.step(action)
        print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}")
    print(f"Battle {battle_idx+1} finished.")
print("All battles finished.")

# find completed battles before loading examples
# online_dset.refresh_files()

