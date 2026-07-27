# inspect_meta.py
import h5py
import cv2
import os

# Point this to one of the hdf5 files you unzipped from EgoDex
hdf5_path = "test/screw_unscrew_fingers_fixture/4.hdf5"
video_path = hdf5_path.replace(".hdf5", ".mp4")

if os.path.exists(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        print("--- 📝 Apple EgoDex Metadata ---")
        # Extract the natural language annotation stored by Apple
        if "llm_description" in f.attrs:
            print(f"Task Description: {f.attrs['llm_description']}")
        if "llm_description2" in f.attrs:
            print(f"Inverse Description: {f.attrs['llm_description2']}")
            
    # Quick frame count check for your interaction boundaries
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    print(f"Total Video Frames: {total_frames}")
else:
    print(f"Could not find the file at {hdf5_path}. Double check your folder structure!")