# serve.py
import sglang as sgl
from sglang.utils import download_and_resize_image
from pydantic import BaseModel
import json
import cv2
import numpy as np

# Import the exact compliance schema we drafted in Phase 1
from src.schema import HandObjectInteraction

def run_egoscribe_inference():
    # 1. Initialize the SGLang local hardware execution engine
    # SGLang automatically targets your CUDA runtime, applying FlashInfer kernel accelerations
    model_path = "XiaomiMiMo/MiMo-VL-7B-RL" 
    
    print("Initializing SGLang Runtime Engine on Lambda Hardware...")
    runtime = sgl.Runtime(
        model_path=model_path,
        tokenizer_path=model_path,
        tp_size=1, # Scale to multi-GPU (e.g., tp_size=2 or 4) if increasing your frame contexts!
        trust_remote_code=True
    )
    sgl.set_default_backend(runtime)
    print("Engine online. System optimized.")

    # 2. Define our SGLang Generation Template Function
    # The 'thinking' capabilities of MiMo-VL are natively captured here
    @sgl.function
    def analyze_industrial_feed(s, video_frames, pydantic_schema_json):
        # Insert the multimodal image sequence into the context state
        s += "You are watching a continuous egocentric industrial work stream.\n"
        for frame in video_frames:
            s += sgl.image(frame)
            
        s += (
            "\nAnalyze this visual chronology. "
            "Show your reasoning step-by-step using <think> tags, evaluating tool placement "
            "and structural changes. Finally, populate the mandatory JSON block matching this exact structure:\n"
            f"{pydantic_schema_json}\n"
        )
        
        # We leverage SGLang's guided decoding engine to guarantee structural syntax alignment
        s += sgl.gen(
            "json_output", 
            max_tokens=1024, 
            temperature=0.0, # Complete determinism for industrial safety auditing
            regex=r"\{.*\}" # Enforces basic JSON bounds
        )

    # 3. Simulate an Egocentric Video Stream Ingestion
    # Open an industrial action clip, sampling frames to pass to the engine
    video_path = "data/samples/sample_assembly_clip.mp4"
    cap = cv2.VideoCapture(video_path)
    sampled_frames = []
    
    # Extract 8 balanced frames from the raw file stream
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, 8, dtype=int)
    
    for idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if idx in frame_indices:
            # Convert default BGR OpenCV array layouts to standard RGB for the VLM
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sampled_frames.append(frame_rgb)
    cap.release()

    # 4. Extract and pass our explicit Pydantic JSON string template
    schema_dump = json.dumps(HandObjectInteraction.model_json_schema(), indent=2)

    # 5. Trigger the Parallel Inference Pipeline Execution
    print("Passing Video Timeline to MiMo-VL-7B-RL + SGLang Guided Backend...")
    state = analyze_industrial_feed.run(
        video_frames=sampled_frames,
        pydantic_schema_json=schema_dump
    )

    # 6. Harvest and display the output
    raw_response = state["json_output"]
    print("\n--- Raw Model Output Stream ---")
    print(raw_response)
    
    # Validate that the generated content strictly parses through our python constraints
    try:
        # Strip out any pre-appended reasoning think blocks if they bleed past regex
        clean_json_str = raw_response[raw_response.find("{"):raw_response.rfind("}")+1]
        validated_data = HandObjectInteraction.parse_raw(clean_json_str)
        print("\n--- 🔍 Validated Compliance Record Generated Successfully ---")
        print(validated_data.json(indent=4))
    except Exception as e:
        print(f"\n Validation Error: Generated output drifted from structural requirements. Error: {e}")

    # Shutdown backend processes cleanly
    runtime.shutdown()

if __name__ == "__main__":
    run_egoscribe_inference()