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

# ADR-0002: the 4 partial-tuning strategies under comparison. Each maps directly onto
# Unsloth's per-component finetune switches, so the training loop below is identical across
# strategies — only this config changes.
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
        finetune_attention_modules=False,
        finetune_mlp_modules=False,
    ),
    "C": dict(  # Language-decoder only — does it live in describing/reasoning over what's seen?
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=False,
        finetune_mlp_modules=False,
    ),
    "D": dict(  # Attention + MLP hybrid — cheaper than A, more expressive than B/C
        finetune_vision_layers=False,
        finetune_language_layers=False,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
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
    return parser.parse_args()


def main():
    args = parse_args()
    strategy_config = STRATEGY_CONFIGS[args.strategy]

    # 1. Initialize the HF Accelerate Environment
    accelerator = Accelerator(gradient_accumulation_steps=2)

    # 2. Load the Processor & Base Model already 4-bit quantized (ADR-0003).
    # This is what makes a run fit a rented 24GB card instead of needing full BF16 headroom.
    model, processor = FastVisionModel.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
    )

    # 3. Apply the strategy-specific partial-tuning LoRA config (ADR-0002).
    model = FastVisionModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        random_state=3407,
        **strategy_config,
    )

    if accelerator.is_main_process:
        model.print_trainable_parameters()

    # 4. Instantiate our Custom Data Pipeline (Using your Toy Sample Subset first!)
    train_dataset = EgocentricHOIDataset(
        json_metadata_path="data/samples/annotations.json",
        video_dir="data/samples/",
        processor=processor,
        num_frames=8
    )

    # Simple, highly explicit collator to gather multimodal dictionaries into uniform batches
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        # Pixel values from video sequences require flexible stacking depending on model implementation
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        image_grid_thw = torch.stack([item["image_grid_thw"] for item in batch]) if "image_grid_thw" in batch[0] else None

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
