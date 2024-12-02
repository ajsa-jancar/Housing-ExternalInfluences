from PIL import Image
import cv2
import sys
import numpy as np
from pathlib import Path

if len(sys.argv) != 2:
    print("Usage: python gif_to_mp4.py <input_gif_path>")
    sys.exit(1)


# Path to the GIF file
gif_path = Path(sys.argv[1])
output_mp4_path = gif_path.with_suffix(".mp4")

# Open the GIF using PIL
gif = Image.open(gif_path)

# Get GIF properties
frame_count = gif.n_frames
frame_width, frame_height = gif.size

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
video_out = cv2.VideoWriter(output_mp4_path, fourcc, 10, (frame_width, frame_height))  # 10 FPS

# Loop through all frames of the GIF
for frame_index in range(frame_count):
    gif.seek(frame_index)  # Move to the next frame
    frame = gif.convert("RGB")  # Convert to RGB (PIL Image)
    frame_np = np.array(frame)  # Convert to NumPy array

    # Convert RGB to BGR (OpenCV expects BGR)
    frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)

    # Write the frame to the video
    video_out.write(frame_bgr)

# Release the video writer
video_out.release()
