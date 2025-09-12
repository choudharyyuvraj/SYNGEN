"""
Custom Dataset Setup Guide for One-DM
====================================

This script helps you set up a custom dataset for handwriting generation.

Required folder structure:
```
custom_styles/
    writer1/
        sample1.png
        sample2.png
        sample3.png
    writer2/
        sample1.png
        sample2.png
    writer3/
        ...
```

Usage Examples:
==============

1. Generate "Hello" in all available writing styles:
   python test_custom.py --style_folder custom_styles --text_source "Hello"

2. Generate multiple words from a comma-separated list:
   python test_custom.py --style_folder custom_styles --text_source "Hello,World,Python,AI"

3. Generate words from a text file:
   python test_custom.py --style_folder custom_styles --text_source words.txt

4. Limit to first 5 texts only:
   python test_custom.py --style_folder custom_styles --text_source "Hello,World,Python,AI,Code,Test" --max_texts 5

Style Image Requirements:
========================
- Format: PNG, JPG, or JPEG
- Grayscale or color (will be converted to grayscale)
- Minimum width: 128 pixels (wider is better)
- Should contain handwritten text samples
- Multiple samples per writer recommended for variety

Text File Format:
================
words.txt can contain:
- One word per line
- Multiple words separated by spaces
- Comma-separated words on one line

Example:
```
Hello
World
Machine Learning
Artificial Intelligence
```

Output:
=======
Generated images will be saved as:
{text}_{writer_id}_{sample_number}.png

For example:
- Hello_writer1_0.png
- Hello_writer2_0.png  
- World_writer1_0.png
"""

import os
import argparse
from PIL import Image
import numpy as np

def create_sample_dataset():
    """Create a sample dataset structure"""
    base_dir = "custom_styles"
    os.makedirs(base_dir, exist_ok=True)
    
    # Create sample writer folders
    for i in range(1, 4):
        writer_dir = os.path.join(base_dir, f"writer{i}")
        os.makedirs(writer_dir, exist_ok=True)
        
        # Create placeholder images
        for j in range(1, 4):
            # Create a simple synthetic handwriting-like image
            img = np.ones((64, 200), dtype=np.uint8) * 255  # White background
            
            # Add some random "handwriting" strokes
            np.random.seed(i * 10 + j)
            for _ in range(20):
                x = np.random.randint(10, 190)
                y = np.random.randint(10, 54)
                w = np.random.randint(5, 30)
                h = np.random.randint(2, 8)
                img[y:y+h, x:x+w] = np.random.randint(0, 100)
            
            # Save as PNG
            pil_img = Image.fromarray(img, mode='L')
            pil_img.save(os.path.join(writer_dir, f"sample{j}.png"))
    
    print(f"Created sample dataset in '{base_dir}' folder")
    print("Replace these placeholder images with real handwriting samples!")

def validate_dataset(style_folder):
    """Validate the custom dataset structure"""
    if not os.path.exists(style_folder):
        print(f"❌ Style folder '{style_folder}' does not exist")
        return False
    
    writers = [w for w in os.listdir(style_folder) 
              if os.path.isdir(os.path.join(style_folder, w))]
    
    if not writers:
        print(f"❌ No writer folders found in '{style_folder}'")
        return False
    
    print(f"✅ Found {len(writers)} writers:")
    
    total_images = 0
    for writer in writers:
        writer_path = os.path.join(style_folder, writer)
        images = [f for f in os.listdir(writer_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not images:
            print(f"  ❌ {writer}: No images found")
        else:
            print(f"  ✅ {writer}: {len(images)} images")
            total_images += len(images)
            
            # Check image dimensions
            for img_file in images[:2]:  # Check first 2 images
                try:
                    img_path = os.path.join(writer_path, img_file)
                    img = Image.open(img_path)
                    width, height = img.size
                    
                    if width < 128:
                        print(f"    ⚠️  {img_file}: Width {width}px is quite narrow (recommend >128px)")
                    else:
                        print(f"    ✅ {img_file}: {width}x{height}px")
                        
                except Exception as e:
                    print(f"    ❌ {img_file}: Error loading - {e}")
    
    print(f"\nTotal: {total_images} images from {len(writers)} writers")
    return total_images > 0

def create_sample_texts():
    """Create sample text files"""
    # Simple words
    with open("simple_words.txt", "w") as f:
        f.write("Hello\nWorld\nPython\nAI\nCode\nData\nTest\nDemo")
    
    # Comma-separated
    with open("comma_words.txt", "w") as f:
        f.write("Machine Learning, Artificial Intelligence, Deep Learning, Neural Networks")
    
    # Sentences
    with open("sentences.txt", "w") as f:
        f.write("Hello World\nMachine Learning\nArtificial Intelligence\nData Science")
    
    print("Created sample text files:")
    print("- simple_words.txt (one word per line)")
    print("- comma_words.txt (comma-separated)")  
    print("- sentences.txt (phrases/sentences)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup custom dataset for One-DM")
    parser.add_argument("--create_sample", action="store_true", 
                       help="Create sample dataset structure")
    parser.add_argument("--validate", type=str, 
                       help="Validate existing dataset folder")
    parser.add_argument("--create_texts", action="store_true",
                       help="Create sample text files")
    
    args = parser.parse_args()
    
    if args.create_sample:
        create_sample_dataset()
    
    if args.validate:
        validate_dataset(args.validate)
    
    if args.create_texts:
        create_sample_texts()
    
    if not any([args.create_sample, args.validate, args.create_texts]):
        print(__doc__)
