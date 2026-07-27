# ADR-0003: Quantization / QLoRA backend

Status: Accepted
Date: 2026-07-24

## Context

`train.py` loads the model in full `bfloat16` (`torch_dtype=torch.bfloat16`, no 4-bit
quantization). With no local GPU and a rented-by-the-hour budget, full BF16 for a 7B model
(~14GB weights alone) doesn't fit free-tier GPUs (Colab T4, 16GB) and eats most of the headroom
on a rented 24GB card, limiting batch size / frame count.

Two options considered:
1. **Unsloth** — wraps `bitsandbytes` 4-bit loading + optimized kernels + the
   `finetune_vision_layers` / `finetune_language_layers` / `finetune_attention_modules` /
   `finetune_mlp_modules` switches that directly implement ADR-0002's strategy set.
2. **Plain bitsandbytes + peft** — what `train.py` already half-uses (`peft` is imported). More
   manual: strategy-specific `target_modules` lists have to be hand-written and verified per
   Qwen2.5-VL module names.

## Decision

Use **Unsloth** for 4-bit QLoRA loading and the per-component finetune switches. It removes an
entire category of manual work (module-name targeting) that would otherwise need to be
correct across all 4 strategies.

## Consequences

- Adds `unsloth` to `requirements.txt`; Unsloth's model support list becomes a real constraint
  on model choice (see [ADR-0001](0001-model-selection.md)).
- `train.py`'s `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)` + manual `LoraConfig`
  block is replaced by Unsloth's `FastVisionModel` (or equivalent) load + `get_peft_model`
  helpers.
- If Unsloth support for the final model choice turns out to be broken/unmaintained, fall back
  to plain `bitsandbytes` + `peft` with hand-written `target_modules` per strategy — more work,
  not a blocker.
