# ADR-0001: Model selection

Status: Proposed (leaning Qwen2.5-VL-7B-Instruct)
Date: 2026-07-24

## Context

`train.py` originally loaded `XiaomiMiMo/MiMo-VL-7B-RL` via
`Qwen2_5_VLForConditionalGeneration` — works because MiMo-VL is built on the Qwen2.5-VL
backbone, but the choice was never deliberate. `serve.py` also targets MiMo-VL via SGLang.

Choosing the QLoRA backend (Unsloth — [ADR-0003](0003-quantization-qlora.md)) narrows this:
Unsloth's best-tested vision fine-tuning support is for Qwen2.5-VL specifically.

## Decision

Lean **Qwen2.5-VL-7B-Instruct** unless a concrete reason to prefer MiMo-VL surfaces
(e.g. MiMo-VL's native `<think>` reasoning traces turn out to matter for the
`point_of_no_return_detected` field). Not fully closed — revisit once Unsloth support for
MiMo-VL is checked directly.

## Consequences

- `serve.py`'s SGLang path and `model_path` will need updating if we switch away from MiMo-VL.
- Community fine-tuning guides/precedent will be easier to find for Qwen2.5-VL.
- If MiMo-VL's `<think>` traces prove valuable later, this decision should be revisited with
  that evidence, not reopened speculatively.
