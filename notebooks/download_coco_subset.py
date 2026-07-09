import os
import urllib.request
import zipfile
import json
import shutil
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

def download_file(url, dest):
    """Download file from URL to destination"""
    if os.path.exists(dest):
        print(f"File already exists: {dest}")
        return
    print(f"Downloading {url} to {dest}")
    urllib.request.urlretrieve(url, dest, reporthook=lambda b, bsize, tsize: print(
        f"Downloaded {b*bsize/tsize*100:.1f}%", end="\r"))
    print()

def prepare_coco_subset(data_dir, classes, max_images_per_class=100, split="val2017"):
    """
    Download and prepare COCO subset for specified classes
    
    Args:
        data_dir: Where to save the dataset
        classes: List of class names to download
        max_images_per_class: Maximum number of images per class
        split: Dataset split ('val2017' or 'train2017')
    """
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    images_dir = os.path.join(data_dir, classes[0], "images")
    masks_dir = os.path.join(data_dir, classes[0], "masks")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(masks_dir, exist_ok=True)
    
    # Download COCO annotations
    annotations_url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    annotations_zip = os.path.join(data_dir, "annotations.zip")
    download_file(annotations_url, annotations_zip)
    
    # Extract annotations if needed
    annotations_dir = os.path.join(data_dir, "annotations")
    if not os.path.exists(annotations_dir):
        print("Extracting annotations...")
        with zipfile.ZipFile(annotations_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Initialize COCO API
    annotations_file = os.path.join(annotations_dir, f"instances_{split}.json")
    coco = COCO(annotations_file)
    
    # Get category IDs for requested classes
    cat_ids = []
    for cls in classes:
        cls_ids = coco.getCatIds(catNms=[cls])
        if cls_ids:
            cat_ids.extend(cls_ids)
        else:
            print(f"Warning: Class '{cls}' not found in COCO")
    
    # Get image IDs containing these categories
    image_ids = []
    for cat_id in cat_ids:
        ids = coco.getImgIds(catIds=[cat_id])
        print(f"Found {len(ids)} images for category {coco.loadCats(cat_id)[0]['name']}")
        # Limit images per class
        if len(ids) > max_images_per_class:
            ids = np.random.choice(ids, max_images_per_class, replace=False).tolist()
        image_ids.extend(ids)
    
    # Remove duplicates
    image_ids = list(set(image_ids))
    print(f"Total unique images to download: {len(image_ids)}")
    
    # Download COCO images archive if not already present
    images_url = f"http://images.cocodataset.org/zips/{split}.zip"
    images_zip = os.path.join(data_dir, f"{split}.zip")
    download_file(images_url, images_zip)
    
    # Extract COCO images archive if needed
    coco_images_dir = os.path.join(data_dir, split)
    if not os.path.exists(coco_images_dir):
        print("Extracting images...")
        with zipfile.ZipFile(images_zip, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    
    # Process each image
    for img_id in tqdm(image_ids, desc="Processing images"):
        # Load image info
        img_info = coco.loadImgs(img_id)[0]
        img_file = img_info['file_name']
        
        # Copy image to our dataset
        src_path = os.path.join(coco_images_dir, img_file)
        dst_path = os.path.join(images_dir, img_file)
        shutil.copy(src_path, dst_path)
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids)
        anns = coco.loadAnns(ann_ids)
        
        # Create mask
        mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
        for ann in anns:
            if ann['category_id'] in cat_ids:
                try:
                    # Convert segmentation to binary mask
                    if isinstance(ann['segmentation'], list):
                        # Some annotations have a list of polygons
                        # Each element might be a polygon (list of points) or something else
                        binary_mask = np.zeros((img_info['height'], img_info['width']), dtype=np.uint8)
                        
                        for seg in ann['segmentation']:
                            # Convert to appropriate format for decode
                            if isinstance(seg, list):  # polygon points
                                # Single polygon - convert to RLE
                                rles = mask_utils.frPyObjects([seg], img_info['height'], img_info['width'])
                                m = mask_utils.decode(rles)
                                if len(m.shape) > 2:
                                    m = m.sum(axis=2) > 0
                                binary_mask = np.maximum(binary_mask, m.astype(np.uint8))
                            elif isinstance(seg, dict) and 'counts' in seg:  # RLE format
                                m = mask_utils.decode(seg)
                                binary_mask = np.maximum(binary_mask, m.astype(np.uint8))
                        
                        # Add to the overall mask
                        mask = np.maximum(mask, binary_mask * 255)
                        
                    elif isinstance(ann['segmentation'], dict) and 'counts' in ann['segmentation']:
                        # RLE format directly
                        m = mask_utils.decode(ann['segmentation'])
                        mask = np.maximum(mask, m.astype(np.uint8) * 255)
                    else:
                        # Skip annotations with unknown segmentation format
                        print(f"Warning: Skipping annotation with unusual segmentation format type: {type(ann['segmentation'])}")
                        continue
                except Exception as e:
                    print(f"Error processing annotation {ann.get('id', 'unknown')}: {str(e)}")
                    continue
        
        # Save mask
        mask_file = img_file.replace('.jpg', '.png')
        mask_path = os.path.join(masks_dir, mask_file)
        Image.fromarray(mask).save(mask_path)
    
    # Create metadata file
    metadata = {
        "classes": classes,
        "num_images": len(image_ids),
        "split": split,
        "source": "COCO"
    }
    with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Dataset ready at {data_dir}")
    print(f"Images: {len(os.listdir(images_dir))}")
    print(f"Masks: {len(os.listdir(masks_dir))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download COCO subset by class")
    parser.add_argument("--output", type=str, default="/home/win11/attention-cim/data/coco_subset", help="Output directory")
    parser.add_argument("--classes", type=str, nargs="+", default=["person", "dog", "cat"], 
                      help="COCO classes to download")
    parser.add_argument("--max-per-class", type=int, default=300, help="Max images per class")
    parser.add_argument("--split", type=str, default="val2017", choices=["val2017", "train2017"],
                      help="Dataset split to use")
    
    args = parser.parse_args()
    
    prepare_coco_subset(args.output, args.classes, args.max_per_class, args.split)
    
    # Visualize a few examples
    images_dir = os.path.join(args.output, args.classes[0],"images")
    if not os.path.exists(images_dir):
        os.makedirs(images_dir, exist_ok=True)
    masks_dir = os.path.join(args.output, args.classes[0], "masks")
    if not os.path.exists(masks_dir):
        os.makedirs(masks_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(images_dir))[:5]
    plt.figure(figsize=(15, 10))
    
    for i, img_file in enumerate(image_files):
        # Load image and mask
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, img_file.replace('.jpg', '.png'))
        
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        # Show image with mask overlay
        plt.subplot(len(image_files), 2, i*2+1)
        plt.imshow(image)
        plt.title(f"Image: {img_file}")
        plt.axis('off')
        
        plt.subplot(len(image_files), 2, i*2+2)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask: {img_file.replace('.jpg', '.png')}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output, "examples.png"))
    plt.close()
    
    print(f"Example visualizations saved to {os.path.join(args.output, 'examples.png')}")