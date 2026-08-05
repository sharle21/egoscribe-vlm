# train.py
import argparse
import torch
from torch.utils.data import DataLoader
from unsloth import FastVisionModel
from accelerate import Accelerator
from tqdm import tqdm

# Import our custom pipeline from our directory layout
from src.dataset import EgocentricHOIDataset

# ADR-0001: Qwen2.5-VL selected — verified Unsloth support for the partial-tuning switches
# ADR-0002 depends on. "unsloth-bnb-4bit" (not plain "bnb-4bit") is Unsloth's Dynamic quant:
# better accuracy for <10% more VRAM.
MODEL_ID = "unsloth/Qwen2.5-VL-7B-Instruct-unsloth-bnb-4bit"

# ADR-0002: the 4 partial-tuning strategies under comparison. Unsloth's 4 switches are TWO
# independent axes, not 4 independent toggles: finetune_{vision,language}_layers picks WHICH
# layers are in scope, finetune_{attention,mlp}_modules picks WHAT within that scope — at least
# one flag on each axis must be True or Unsloth raises "No modules to finetune" (caught during
# the Colab smoke test: B and D's original configs each zeroed out one whole axis).
STRATEGY_CONFIGS = {
    "A": dict(  # Full LoRA — baseline / upper bound on trainable params
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    ),
    "B": dict(  # Vision-encoder only — does the domain gap live in *seeing* egocentric frames?
        finetune_vision_layers=True,
        finetune_language_layers=False,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    ),
    "C": dict(  # Language-decoder only — does it live in describing/reasoning over what's seen?
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
    ),
    "D": dict(  # Attention-only, both modalities — cheaper than A (no MLP), broader than B/C
        finetune_vision_layers=True,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=False,
    ),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGY_CONFIGS),
        required=True,
        help="Which partial-tuning strategy to run (see ADR-0002).",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_frames", type=int, default=8, help="Lower for small-VRAM smoke tests")
    parser.add_argument(
        "--gradient_checkpointing", type=str, default="unsloth", choices=["unsloth", "true", "false"],
        help="'false' works around a dtype-mismatch bug (frozen vision layer weight vs "
             "activation dtype) that only surfaces when a whole component is untouched by LoRA "
             "(e.g. Strategy C) — costs more VRAM since it disables the forward-recompute path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    strategy_config = STRATEGY_CONFIGS[args.strategy]

    # 1. Initialize the HF Accelerate Environment
    accelerator = Accelerator(gradient_accumulation_steps=2)

    # 2. Load the Processor & Base Model already 4-bit quantized (ADR-0003).
    # This is what makes a run fit a rented 24GB card instead of needing full BF16 headroom.
    # dtype must be picked explicitly, not left to Unsloth's default: on hardware without real
    # bf16 tensor cores (e.g. T4, compute capability 7.5), the 4-bit weights' dequant compute
    # dtype can end up bf16 while frozen/untouched layers' activations fall back to float32 —
    # this only bites strategies that leave a whole component untouched (e.g. Strategy C,
    # vision layers frozen with no LoRA), since layers WITH LoRA adapters get a consistent
    # cast path forced on them regardless.
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        dtype=compute_dtype,
    )

    # 3. Apply the strategy-specific partial-tuning LoRA config (ADR-0002).
    gc_arg = {"unsloth": "unsloth", "true": True, "false": False}[args.gradient_checkpointing]
    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        random_state=3407,
        use_gradient_checkpointing=gc_arg,
        **strategy_config,
    )

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # 4. Instantiate our Custom Data Pipeline (Using your Toy Sample Subset first!)
    train_dataset = EgocentricHOIDataset(
        json_metadata_path="data/samples/annotations.json",
        video_dir="data/samples/",
        processor=processor,
        num_frames=args.num_frames
    )

    # Simple, highly explicit collator to gather multimodal dictionaries into uniform batches
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        # Qwen2.5-VL flattens all image patches across the whole batch into one tensor
        # (images can have different patch counts), with image_grid_thw tracking per-image
        # grid dims in that same flat ordering — so these must be concatenated, not stacked
        # into a new batch dimension, or grid_thw.tolist() unpacks the wrong shape downstream.
        pixel_values = torch.cat([item["pixel_values"] for item in batch], dim=0)
        image_grid_thw = torch.cat([item["image_grid_thw"] for item in batch], dim=0) if "image_grid_thw" in batch[0] else None

        batch_dict = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": pixel_values
        }
        if image_grid_thw is not None:
            batch_dict["image_grid_thw"] = image_grid_thw
        return batch_dict

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    # 5. Optimization & Learning Schedule setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # 6. Prepare Everything via HF Accelerate
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # 7. The Core Training Loop — identical across strategies A-D by construction
    model.train()

    for epoch in range(args.epochs):
        total_loss = 0
        progress_bar = tqdm(
            train_dataloader,
            desc=f"[{args.strategy}] Epoch {epoch+1}",
            disable=not accelerator.is_local_main_process
        )

        for batch in progress_bar:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss

                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})

        if accelerator.is_main_process:
            print(f"[{args.strategy}] Epoch {epoch+1} finished. Average Loss: {total_loss / len(train_dataloader):.4f}")

    # 8. Un-wrap and Save the Finetuned Adapters, namespaced by strategy so runs don't collide
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        output_dir = f"./saved_egoscribe_adapters/strategy_{args.strategy}"
        unwrapped_model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"Saved strategy {args.strategy} adapters to {output_dir}")

if __name__ == "__main__":
    main()
