import torch
import torch.nn as nn
import os
import json
import requests
from ibm_watson_machine_learning import APIClient
from requests.exceptions import RequestException

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

        # API Key setup for IBM Watson
        API_KEY = "ZRoHe-s7mG7_l0I4AvI_RSdVxUsLdLB1GUQDAq5HWMtW"
        # Get IAM token
        token_resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
                "apikey": API_KEY,
            },
        )
        IAM_TOKEN = self.get_iam_token()

        # IBM Watson config
        self.watson_url = "https://us-south.ml.cloud.ibm.com/ml/v1/deployments/pokemon_action_llm_v01/text/generation_stream?version=2021-05-01"
        self.watson_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {IAM_TOKEN}",
        }

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
    def get_iam_token(self):
      """Fetch a fresh IAM token from IBM Cloud."""
      token_resp = requests.post(
          "https://iam.cloud.ibm.com/identity/token",
          headers={"Content-Type": "application/x-www-form-urlencoded"},
          data={
              "grant_type": "urn:ibm:params:oauth:grant-type:apikey",
              "apikey": "ZRoHe-s7mG7_l0I4AvI_RSdVxUsLdLB1GUQDAq5HWMtW",
          },
      )
      token_resp.raise_for_status()
      return token_resp.json()["access_token"]

    def call_watson_llm(self, text: str) -> int:
        """Send a single battle state to Watson LLM and return an action id (0–7)."""
        payload = {
            "input": f"""Game Rules:
                    1. Each player has a team of up to 6 Pokémon.
                    2. Only one Pokémon is active at a time. The active Pokémon can use one of up to 4 moves per turn, or be switched out for a teammate.
                    3. Moves have a type (e.g., Fire, Water), category (Physical, Special, Status), and may have effects (e.g., burn, paralysis).
                    4. Switching to another Pokémon counts as a turn and can change type advantages and statuses.
                    5. Some Pokémon can Tera-evolve, which modifies their type for one turn. Using Tera is optional.
                    6. Status conditions (paralysis, sleep, burn, freeze, poison) and stat changes (boosts/reductions) affect outcomes.
                    7. A move or switch is invalid if the Pokémon cannot perform it (e.g., fainted, locked into another move).

                    Action Space:
                    - Use one of the available moves: "use move {{move_name}}"
                    - Switch to a teammate: "switch to {{pokemon_name}}"
                    - Optionally append "[Tera]" to a move if the Pokémon wants to use its Tera-evolution.

                    Task:
                    Given a detailed description of the current battle state, including the active Pokémon, opponent, available moves, move effects, statuses, and teammates, suggest the single best valid action from the action space above. 
                    Respond in the format: "use move 1" or "switch to slot 2". Always ensure that your suggested action is valid.

                    Battle state:
                    {text}"""
        }

        for attempt in range(3):
          try:
              with requests.post(
                  self.watson_url,
                  headers=self.watson_headers,
                  json=payload,
                  stream=True,
                  timeout=(5, 60)  # 5s connect, 60s read
              ) as resp:
                  if resp.status_code == 401:
                      print("🔑 Token expired, refreshing...")
                      self.watson_headers["Authorization"] = f"Bearer {self.get_iam_token()}"
                      continue  # retry after refresh

                  resp.raise_for_status()

                  output = []
                  for line in resp.iter_lines():
                      if not line:
                          continue
                      text_line = line.decode("utf-8")
                      if text_line.startswith("data: "):
                          try:
                              data = json.loads(text_line[len("data: "):])
                              results = data.get("results", [])
                              if results:
                                  output.append(
                                      results[0].get("generated_text", "").strip()
                                  )
                          except json.JSONDecodeError:
                              pass

                  action_text = " ".join(output).lower().strip()
                  return self.action_map.get(action_text, 0)

          except RequestException as e:
              print(f"⚠️ Request failed (attempt {attempt+1}): {e}")
              if attempt == 2:  # after 3 tries
                  raise


    def forward(self, text_batch, num_batch):
        """
        text_batch: list of battle states (strings)
        num_batch: tensor [batch_size, num_dim]
        """
        device = num_batch.device

        # Convert LLM outputs → action ids
        action_ids = [self.call_watson_llm(txt) for txt in text_batch]
        action_ids = torch.tensor(action_ids, dtype=torch.long, device=device)

        # Encode numbers
        num_emb = self.num_encoder(num_batch)

        # Encode action embeddings
        action_emb = self.action_embedding(action_ids)

        # Fuse numeric + LLM signals
        fused = torch.cat([num_emb, action_emb], dim=-1)
        logits = self.fusion(fused)

        return logits
