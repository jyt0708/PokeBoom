import sys
import os

# Ensure the inner metamon/ (the actual package) is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "metamon"))

from metamon.env import get_metamon_teams
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace

print("1")
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
obs_seq, action_seq, reward_seq, done_seq = offline_dset[0]
print(f"Offline ds length: {len(offline_dset)}")
print(f"Observation sequence: {obs_seq}")
print(f"Action sequence: {action_seq}")
print(f"Reward sequence: {reward_seq}")