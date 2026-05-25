"""Policy training — GRPO + LoRA recipe ported from PharmaRL.

To be implemented in Phase 2. Same recipe template:
  base:    Llama-3.2-3B-Instruct (4-bit NF4) or ESM-3
  PEFT:    LoRA r=16, α=32, all attn/MLP projections
  RL:      GRPO group size G=8, K3 KL anchor β=0.04, lr=5e-6, AdamW
  serving: Unsloth FastLanguageModel
"""
