import os
import argparse
import glob
import numpy as np
import imageio
from tqdm import tqdm


def sort_naturally(file_list):
    """Sort files in natural order (e.g., img1, img2, img10 instead of img1, img10, img2)."""
    import re
    def extract_number(filename):
        numbers = re.findall(r'\d+', os.path.basename(filename))
        return int(numbers[0]) if numbers else 0
    return sorted(file_list, key=extract_number)


def create_video_from_images(image_dir, output_path, fps=30, quality=8, ext='*.png'):
    """
    Create a video from a directory of images.
    
    Args:
        image_dir: Directory containing images
        output_path: Path to save the output video
        fps: Frames per second
        quality: Video quality (0-10, 10 being highest)
        ext: File extension pattern to match
    """
    # Find all images with the specified extension
    image_paths = glob.glob(os.path.join(image_dir, ext))
    
    # If no images found with the specified extension, try other common extensions
    if not image_paths:
        for possible_ext in ['*.jpg', '*.jpeg', '*.PNG', '*.JPEG', '*.JPG']:
            if possible_ext != ext:
                print(f"No images found with extension {ext}, trying {possible_ext}")
                image_paths = glob.glob(os.path.join(image_dir, possible_ext))
                if image_paths:
                    break
    
    if not image_paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    
    # Sort images in natural order
    image_paths = sort_naturally(image_paths)
    print(f"Found {len(image_paths)} images")
    
    # Read the first image to get dimensions
    first_img = imageio.imread(image_paths[0])
    h, w = first_img.shape[:2]
    
    print(f"Creating video with dimensions {w}x{h}")
    print(f"Output file: {output_path}")
    
    # Ensure output has a proper video extension
    _, ext = os.path.splitext(output_path)
    if ext.lower() not in ['.mp4', '.avi', '.mov', '.mkv']:
        print(f"Warning: {ext} may not be a valid video format. Recommended: .mp4")
    
    # Create a writer object with explicit format to avoid TIFF writer issues
    writer = imageio.get_writer(output_path, format='FFMPEG', fps=fps, quality=quality)
    
    # Add all images to the video
    for img_path in tqdm(image_paths, desc="Processing images"):
        img = imageio.imread(img_path)
        writer.append_data(img)
    
    writer.close()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a video from a sequence of images")
    parser.add_argument("--image_dir", type=str, default="test-images", 
                        help="Directory containing images (default: test-images)")
    parser.add_argument("--output", type=str, default="output_video.mp4", 
                        help="Output video filename (default: output_video.mp4)")
    parser.add_argument("--fps", type=int, default=30, 
                        help="Frames per second (default: 30)")
    parser.add_argument("--quality", type=int, default=8, choices=range(11), 
                        help="Video quality 0-10 (default: 8)")
    parser.add_argument("--ext", type=str, default="*.png", 
                        help="Image file extension (default: *.png)")
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    create_video_from_images(
        args.image_dir, 
        args.output, 
        fps=args.fps, 
        quality=args.quality, 
        ext=args.ext
    )