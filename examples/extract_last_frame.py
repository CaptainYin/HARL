#!/usr/bin/env python3
"""
Script to extract the last frame of a GIF file and save it as an image.
"""

import argparse
import os
import sys
from PIL import Image, ImageSequence


def extract_last_frame(gif_path, output_path=None, output_format='PNG'):
    """
    Extract the last frame from a GIF file.
    
    Args:
        gif_path (str): Path to the input GIF file
        output_path (str, optional): Path for the output image. If None, auto-generate
        output_format (str): Output format ('PNG', 'JPEG', 'BMP', etc.)
    
    Returns:
        str: Path to the saved image file
    """
    
    if not os.path.exists(gif_path):
        raise FileNotFoundError(f"GIF file not found: {gif_path}")
    
    # Open the GIF file
    try:
        with Image.open(gif_path) as gif:
            # Check if it's actually a GIF
            if gif.format != 'GIF':
                print(f"Warning: File format is {gif.format}, not GIF")
            
            # Get all frames
            frames = []
            for frame in ImageSequence.Iterator(gif):
                frames.append(frame.copy())
            
            if not frames:
                raise ValueError("No frames found in the GIF")
            
            # Get the last frame
            last_frame = frames[-1]
            
            # Convert to RGB if saving as JPEG (JPEG doesn't support transparency)
            if output_format.upper() == 'JPEG' and last_frame.mode in ('RGBA', 'LA', 'P'):
                last_frame = last_frame.convert('RGB')
            
            # Generate output path if not provided
            if output_path is None:
                base_name = os.path.splitext(os.path.basename(gif_path))[0]
                output_path = f"{base_name}_last_frame.{output_format.lower()}"
            
            # Save the last frame
            last_frame.save(output_path, format=output_format)
            
            print(f"Successfully extracted last frame from GIF")
            print(f"Total frames in GIF: {len(frames)}")
            print(f"Last frame saved to: {output_path}")
            print(f"Frame size: {last_frame.size}")
            print(f"Frame mode: {last_frame.mode}")
            
            return output_path
            
    except Exception as e:
        raise Exception(f"Error processing GIF: {str(e)}")


def extract_all_frames(gif_path, output_dir=None, output_format='PNG'):
    """
    Extract all frames from a GIF file (bonus function).
    
    Args:
        gif_path (str): Path to the input GIF file
        output_dir (str, optional): Directory for output images
        output_format (str): Output format
    
    Returns:
        list: List of saved frame paths
    """
    
    if not os.path.exists(gif_path):
        raise FileNotFoundError(f"GIF file not found: {gif_path}")
    
    # Create output directory if not provided
    if output_dir is None:
        base_name = os.path.splitext(os.path.basename(gif_path))[0]
        output_dir = f"{base_name}_frames"
    
    os.makedirs(output_dir, exist_ok=True)
    
    saved_frames = []
    
    try:
        with Image.open(gif_path) as gif:
            for i, frame in enumerate(ImageSequence.Iterator(gif)):
                frame_copy = frame.copy()
                
                # Convert to RGB if saving as JPEG
                if output_format.upper() == 'JPEG' and frame_copy.mode in ('RGBA', 'LA', 'P'):
                    frame_copy = frame_copy.convert('RGB')
                
                # Save each frame
                frame_path = os.path.join(output_dir, f"frame_{i:04d}.{output_format.lower()}")
                frame_copy.save(frame_path, format=output_format)
                saved_frames.append(frame_path)
            
            print(f"Extracted {len(saved_frames)} frames to: {output_dir}")
            return saved_frames
            
    except Exception as e:
        raise Exception(f"Error extracting all frames: {str(e)}")


def get_gif_info(gif_path):
    """
    Get information about a GIF file.
    
    Args:
        gif_path (str): Path to the GIF file
    
    Returns:
        dict: Information about the GIF
    """
    
    try:
        with Image.open(gif_path) as gif:
            frames = []
            for frame in ImageSequence.Iterator(gif):
                frames.append(frame)
            
            info = {
                'format': gif.format,
                'size': gif.size,
                'mode': gif.mode,
                'num_frames': len(frames),
                'duration': getattr(gif, 'info', {}).get('duration', 'Unknown'),
                'loop': getattr(gif, 'info', {}).get('loop', 'Unknown'),
                'filename': os.path.basename(gif_path)
            }
            
            return info
            
    except Exception as e:
        raise Exception(f"Error getting GIF info: {str(e)}")


def main():
    parser = argparse.ArgumentParser(description='Extract the last frame from a GIF file')
    parser.add_argument('gif_file', help='Path to the input GIF file')
    parser.add_argument('-o', '--output', help='Output file path (optional)')
    parser.add_argument('-f', '--format', default='PNG', choices=['PNG', 'JPEG', 'BMP', 'TIFF'],
                       help='Output image format (default: PNG)')
    parser.add_argument('--all-frames', action='store_true', 
                       help='Extract all frames instead of just the last one')
    parser.add_argument('--info', action='store_true', 
                       help='Show GIF information without extracting frames')
    
    args = parser.parse_args()
    
    try:
        # Show GIF info if requested
        if args.info:
            info = get_gif_info(args.gif_file)
            print(f"\nGIF Information:")
            print(f"  File: {info['filename']}")
            print(f"  Format: {info['format']}")
            print(f"  Size: {info['size'][0]}x{info['size'][1]} pixels")
            print(f"  Mode: {info['mode']}")
            print(f"  Frames: {info['num_frames']}")
            print(f"  Duration: {info['duration']} ms per frame")
            print(f"  Loop: {info['loop']}")
            return 0
        
        # Extract all frames if requested
        if args.all_frames:
            frames = extract_all_frames(args.gif_file, output_format=args.format)
            print(f"Last frame is: {frames[-1]}")
            return 0
        
        # Extract just the last frame (default behavior)
        output_path = extract_last_frame(args.gif_file, args.output, args.format)
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
