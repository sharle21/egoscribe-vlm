# src/dataset.py
import os
import torch
from torch.utils.data import Dataset
from decord import VideoReader, cpu
import json

class EgocentricHOIDataset(Dataset):
    def __init__(self, json_metadata_path, video_dir, processor, num_frames=8):
        """
        Args:
            json_metadata_path: Path to annotations mapping videos to actions & state changes
            video_dir: Directory containing raw .mp4 clips
            processor: The VLM Processor (e.g., Qwen2.5-VL or Llama-3.2-Vision processor)
            num_frames: Number of dense frames to sample per clip
        """
        with open(json_metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.video_dir = video_dir
        self.processor = processor
        self.num_frames = num_frames

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
        prompt = (
            "<image>\nAnalyze this egocentric video sequence. "
            "Identify the active tool, target object, action verb, "
            "and whether a permanent point-of-no-return state change has occurred. "
            "Output your final answer strictly adhering to this JSON schema."
        )
        
        # 3. Format the data using the VLM's native multimodal processor
        # This converts text to token IDs and frames into normalized pixel tensors
        inputs = self.processor(
            text=prompt,
            images=list(video_frames), # Processes the sequence of sampled frames
            padding="max_length",
            max_length=512,
            return_tensors="pt"
        )
        
        # Remove the batch dimension added by the processor default wrapper
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # 4. Attach the ground truth text label (the expected JSON output string)
        # Pad positions are masked to -100 so cross-entropy loss ignores them; the real
        # output JSON is usually much shorter than max_length, so without this most of the
        # loss signal would come from padding tokens instead of the actual answer.
        labels = self.processor.tokenizer(
            json.dumps(item["expected_output"]),
            padding="max_length",
            max_length=256,
            return_tensors="pt"
        )["input_ids"].squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        inputs["labels"] = labels
        return inputs