# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import json

class EgocentricHOIDataset(Dataset):
    def __init__(self, json_metadata_path, video_dir, processor, num_frames=8,
                 min_pixels=64 * 28 * 28, max_pixels=256 * 28 * 28, max_length=2560):
        """
        Args:
            json_metadata_path: Path to annotations mapping videos to actions & state changes
            video_dir: Directory containing raw .mp4 clips
            processor: The VLM Processor (e.g., Qwen2.5-VL or Llama-3.2-Vision processor)
            num_frames: Number of dense frames to sample per clip
            min_pixels/max_pixels: caps per-frame resolution passed to the processor. Qwen2.5-VL
                vision attention cost scales with patch count (quadratic), and native/full-res
                video frames times num_frames patches easily OOMs a 16GB card — these defaults
                are intentionally small for budget GPUs; raise them if VRAM allows.
            max_length: fixed length of the final (prompt + answer) sequence used for padding
                each item to a uniform batchable size. Must comfortably exceed the real
                image-expanded prompt length (num_frames * ~tokens-per-frame, which varies with
                min/max_pixels) plus the JSON answer length — raising num_frames/max_pixels
                without raising this will hit the ValueError below.
        """
        with open(json_metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.max_length = max_length

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        item = self.metadata[idx]
        video_path = os.path.join(self.video_dir, item["video_file"])
        
        # 1. High-speed video frame loading using Decord
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        # Target the specific window where the hand-object interaction occurs
        start_frame = item["interaction_start_frame"]
        end_frame = item["interaction_end_frame"]
        
        # Linearly sample 'num_frames' between the start and end of the action
        frame_indices = torch.linspace(start_frame, min(end_frame, total_frames - 1), self.num_frames).long().tolist()
        
        # Extract frames as a numpy array / torch tensor [Num_Frames, H, W, C]
        video_frames = vr.get_batch(frame_indices).asnumpy()
        
        # 2. Build the system prompt forcing the JSON schema
        # NOTE: must go through apply_chat_template, not a hand-written "<image>\n" string —
        # that literal text isn't a real special token, so the processor never inserts actual
        # image-placeholder tokens and the vision features have nowhere to slot in
        # (fails downstream with "Image features and image tokens do not match").
        # apply_chat_template emits one placeholder per {"type": "image"} entry, which the
        # processor then expands to match each frame's real patch count.
        messages = [{
            "role": "user",
            "content": [{"type": "image"} for _ in range(len(video_frames))] + [{
                "type": "text",
                "text": (
                    "Analyze this egocentric video sequence. "
                    "Identify the active tool, target object, action verb, "
                    "and whether a permanent point-of-no-return state change has occurred. "
                    "Output your final answer strictly adhering to this JSON schema."
                )
            }]
        }]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 3. Format the prompt+images using the VLM's native multimodal processor.
        # No padding/truncation here — this must return the REAL image-expanded length so we
        # can concatenate it with the answer below; padding happens once, after concatenation.
        prompt_inputs = self.processor(
            text=prompt,
            images=list(video_frames), # Processes the sequence of sampled frames
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            return_tensors="pt"
        )
        prompt_ids = prompt_inputs["input_ids"][0]
        prompt_attn = prompt_inputs["attention_mask"][0]

        # 4. Tokenize the ground-truth JSON answer and append an EOS so the model learns to stop.
        answer_ids = self.processor.tokenizer(
            json.dumps(item["expected_output"]),
            add_special_tokens=False,
            return_tensors="pt"
        )["input_ids"][0]
        eos_id = self.processor.tokenizer.eos_token_id
        answer_ids = torch.cat([answer_ids, torch.tensor([eos_id], dtype=answer_ids.dtype)])

        # 5. Concatenate prompt + answer into ONE causal-LM training sequence. labels must be
        # the same length as input_ids (one label per input position) — masking the prompt
        # portion with -100 means loss is only computed on the answer tokens, not on the
        # question/images the model is conditioning on.
        input_ids = torch.cat([prompt_ids, answer_ids])
        attention_mask = torch.cat([prompt_attn, torch.ones_like(answer_ids)])
        labels = torch.cat([torch.full_like(prompt_ids, -100), answer_ids.clone()])

        seq_len = input_ids.shape[0]
        if seq_len > self.max_length:
            raise ValueError(
                f"{item['video_file']}: prompt+answer length {seq_len} exceeds max_length="
                f"{self.max_length}. Reduce num_frames/max_pixels or raise max_length."
            )
        pad_len = self.max_length - seq_len
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids = torch.cat([input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)])
        attention_mask = torch.cat([attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)])
        labels = torch.cat([labels, torch.full((pad_len,), -100, dtype=labels.dtype)])

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": prompt_inputs["pixel_values"],
        }
        if "image_grid_thw" in prompt_inputs:
            inputs["image_grid_thw"] = prompt_inputs["image_grid_thw"]
        return inputs