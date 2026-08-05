# ADR-0001: Model selection

Status: Accepted
Date: 2026-07-29

## Context

`train.py` originally loaded `XiaomiMiMo/MiMo-VL-7B-RL` via
`Qwen2_5_VLForConditionalGeneration` — works because MiMo-VL is built on the Qwen2.5-VL
backbone, but the choice was never deliberate. `serve.py` also targets MiMo-VL via SGLang.

Choosing the QLoRA backend (Unsloth — [ADR-0003](0003-quantization-qlora.md)) narrows this:
Unsloth's best-tested vision fine-tuning support is for Qwen2.5-VL specifically. Checked
web-search-confirmed evidence (2026-07-28): `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit`
is a real, published model on HuggingFace, and Unsloth's vision fine-tuning docs explicitly
document the `finetune_vision_layers`/`finetune_language_layers`/`finetune_attention_modules`/
`finetune_mlp_modules` switches ADR-0002's strategy set depends on, for this model. Newer
Qwen3-VL/Qwen3.5-VL models are also Unsloth-supported in 2026, but the same partial-tuning
switches were not confirmed documented for them specifically — since the whole ablation
methodology depends on those switches working per-strategy, the less-verified newer models
weren't worth the risk for this project.

## Decision

**Qwen2.5-VL-7B-Instruct**, loaded via `unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit`
(Unsloth's "Dynamic" 4-bit quant — better accuracy than plain `bnb-4bit` for <10% more VRAM).
Not MiMo-VL — dropped since it would mean re-verifying Unsloth's partial-tuning switch support
from scratch with less community precedent, for a project already convergent on Qwen2.5-VL.

## Consequences

- `serve.py`'s SGLang path and `model_path` will need updating if we switch away from MiMo-VL.
- Community fine-tuning guides/precedent will be easier to find for Qwen2.5-VL.
- If MiMo-VL's `<think>` traces prove valuable later, this decision should be revisited with
  that evidence, not reopened speculatively.
