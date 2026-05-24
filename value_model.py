"""
Proof State Value Model.

Llama-3.2-1B with a scalar value head.
Takes a proof state, outputs P(state leads to completed proof) in [0, 1].
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class ProofValueModel(nn.Module):
    """Predicts discounted distance to proof completion.

    Output is gamma^d where d is estimated remaining steps to QED.
    Values near 1.0 = close to done, near 0.0 = far away or dead end.
    """

    def __init__(self, base_model_path="meta-llama/Llama-3.2-1B", freeze_backbone=True):
        super().__init__()
        self.backbone = AutoModelForCausalLM.from_pretrained(
            base_model_path, torch_dtype=torch.bfloat16
        )
        hidden_size = self.backbone.config.hidden_size  # 2048 for 1B
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Value head in float32 for stability
        self.value_head = self.value_head.float()

    def forward(self, input_ids, attention_mask):
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        last_hidden = outputs.hidden_states[-1]  # (B, seq_len, hidden)

        # Get last non-padding token's hidden state
        seq_lens = attention_mask.sum(dim=1) - 1
        last_token_hidden = last_hidden[torch.arange(len(seq_lens)), seq_lens]

        # sigmoid outputs [0, 1] — represents gamma^d (discounted distance)
        value = torch.sigmoid(self.value_head(last_token_hidden.float()))
        return value.squeeze(-1)  # (B,)


class ValueModelServer:
    """Simple inference server for the value model."""

    def __init__(self, model_path, device="cuda:0", max_length=2048):
        self.device = device
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = ProofValueModel(model_path, freeze_backbone=True)
        # Load value head weights if they exist
        import os
        value_head_path = os.path.join(model_path, "value_head.pt")
        if os.path.exists(value_head_path):
            self.model.value_head.load_state_dict(torch.load(value_head_path))

        self.model = self.model.to(device).eval()

    @torch.no_grad()
    def score(self, proof_states: list[str]) -> list[float]:
        """Score a batch of proof states. Returns values in [0, 1]."""
        inputs = self.tokenizer(
            proof_states,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

        values = self.model(inputs.input_ids, inputs.attention_mask)
        return values.cpu().tolist()
