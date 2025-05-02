#!/bin/bash

# Simple shell script to create a video from test images

# Default parameters
IMAGE_DIR="./test_images/test_transformer"
OUTPUT="output_video2.mp4"
FPS=30
QUALITY=8
EXT="*.png"

# Run the Python script
echo "Creating video from images in $IMAGE_DIR..."
python render-video.py --image_dir "$IMAGE_DIR" --output "$OUTPUT" --fps "$FPS" --quality "$QUALITY" --ext "$EXT"