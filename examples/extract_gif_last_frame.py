#!/usr/bin/env python3
"""
Simple script to extract the last frame of a GIF file.
Usage: python extract_gif_last_frame.py <gif_file> [output_file]
"""

import sys
import os

try:
    from PIL import Image, ImageSequence
except ImportError:
    print("Error: Pillow library not found. Please install it with:")
    print("pip install Pillow")
    sys.exit(1)


def extract_last_frame_simple(gif_path, output_path=None):
    """
    Simple function to extract the last frame from a GIF.
    """
    
    if not os.path.exists(gif_path):
        print(f"Error: GIF file '{gif_path}' not found")
        return False
    
    try:
        # Open the GIF
        with Image.open(gif_path) as gif:
            # Get all frames
            frames = list(ImageSequence.Iterator(gif))
            
            if not frames:
                print("Error: No frames found in the GIF")
                return False
            
            # Get the last frame
            last_frame = frames[-1].copy()
            
            # Generate output filename if not provided
            if output_path is None:
                base_name = os.path.splitext(gif_path)[0]
                output_path = f"{base_name}_last_frame.png"
            
            # Save the last frame
            last_frame.save(output_path, 'PNG')
            
            print(f"✓ Extracted last frame from '{gif_path}'")
            print(f"✓ Total frames: {len(frames)}")
            print(f"✓ Saved to: '{output_path}'")
            print(f"✓ Frame size: {last_frame.size}")
            
            return True
            
    except Exception as e:
        print(f"Error processing GIF: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: python extract_gif_last_frame.py <gif_file> [output_file]")
        print("Example: python extract_gif_last_frame.py animation.gif")
        print("Example: python extract_gif_last_frame.py animation.gif last_frame.png")
        return 1
    
    gif_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = extract_last_frame_simple(gif_file, output_file)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
