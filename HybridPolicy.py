import torch
import torch.nn as nn
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