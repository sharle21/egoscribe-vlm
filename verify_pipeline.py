# verify_pipeline.py
import os
from transformers import AutoProcessor
from src.dataset import EgocentricHOIDataset

def run_pipeline_check():
    print("🔮 Initializing MiMo-VL Multi-Modal Processor Hook...")
    model_id = "XiaomiMiMo/MiMo-VL-7B-RL"
    
    # Initialize the specific vision processor from the hub config
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    
    print("📁 Checking local configuration files...")
    anno_path = "data/annotations.json"
    video_dir = "data/samples/"
    
    if not os.path.exists(anno_path):
        print(f"❌ Aborting! Cannot locate data array: {anno_path}")
        return
        
    print("⚡ Constructing custom Egocentric Data Loader instance...")
    dataset = EgocentricHOIDataset(json_metadata_path=anno_path,video_dir=video_dir, processor=processor, num_frames=8)
    
    print(f"📊 Validated Dataset Dimensions: Found {len(dataset)} targeted video rows.")
    
    # Pull sample index 0 directly to monitor performance metrics
    print("\n📦 Pulling Sample index 0 through data pipeline tensors...")
    try:
        sample = dataset[0]
        
        print("\n--- Tensor Verification Report ---")
        for key, val in sample.items():
            if hasattr(val, "shape"):
                print(f"🔹 Tensor Matrix '{key}': Shape -> {list(val.shape)}")
            else:
                print(f"🔹 Metadata Field '{key}': Type -> {type(val)}")
                
        print("\n✅ Success! EgoDex videos map to compliance models cleanly.")
        
    except Exception as e:
        print(f"\n❌ Pipeline Breakdown Detected: {str(e)}")

if __name__ == "__main__":
    run_pipeline_check()