#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import yaml
from parsers.pdf_parser import render_page_to_image, extract_text_with_pymupdf
from detectors.vision_detectors import load_block_detector, load_table_detector, detect_blocks
from fusion.fusion import merge_boxes
from utils.output import visualize_page

def test_comprehensive_deduplication():
    """Test the comprehensive image deduplication across PDF native and vision detections"""
    
    # Load configuration
    with open('src/configs/models.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("======================================================================")
    print("TESTING COMPREHENSIVE IMAGE DEDUPLICATION")
    print("======================================================================")
    
    pdf_path = "sample.pdf"
    page_num = 0  # Test on page 0 which has the Baker Hughes logo
    
    print(f"Testing comprehensive deduplication on {pdf_path}, page {page_num}")
    
    # Parse PDF
    pdf_boxes = extract_text_with_pymupdf(pdf_path, page_num, dpi=config.get('render_dpi', 300))
    page_image, _ = render_page_to_image(pdf_path, page_num, dpi=config.get('render_dpi', 300))
    
    print(f"✅ PDF Analysis:")
    text_elements = [p for p in pdf_boxes if p.get('type') == 'text']
    image_elements = [p for p in pdf_boxes if p.get('type') == 'image']
    print(f"   Text elements: {len(text_elements)}")
    print(f"   PDF native images: {len(image_elements)}")
    
    # Load models
    print("✅ Loading vision models...")
    block_proc, block_model = load_block_detector(config['block_detector']['model_name'])
    
    # Vision detection
    print("✅ Running vision detection...")
    vision_boxes = detect_blocks(page_image, block_proc, block_model, 
                                threshold=config['block_detector']['confidence_threshold'])
    
    print(f"✅ Vision detection results:")
    vision_images = [v for v in vision_boxes if v['label'].lower() in ['picture', 'image', 'figure']]
    print(f"   Vision detected images: {len(vision_images)}")
    print(f"   Total images before deduplication: {len(image_elements) + len(vision_images)}")
    
    # Show details of detected images
    print("\n=== IMAGE DETECTIONS BEFORE DEDUPLICATION ===")
    for i, img in enumerate(image_elements):
        print(f"PDF Image {i+1}: bbox={img['bbox']}")
    
    for i, img in enumerate(vision_images):
        print(f"Vision Image {i+1}: bbox={img['bbox']}, score={img.get('score', 0):.3f}")
    
    # Merge with comprehensive deduplication
    print("\n✅ Applying comprehensive deduplication...")
    merged_results = merge_boxes(pdf_boxes, vision_boxes, config=config)
    
    # Analyze results
    final_images = [elem for elem in merged_results if elem['label'].lower() in ['picture', 'image']]
    final_text = [elem for elem in merged_results if elem['label'].lower() in ['text', 'title', 'header']]
    
    print(f"\n✅ Final results after comprehensive deduplication:")
    print(f"   Images: {len(final_images)}")
    print(f"   Text elements: {len(final_text)}")
    print(f"   Total elements: {len(merged_results)}")
    
    # Show details of final images
    print("\n=== FINAL IMAGE DETECTIONS ===")
    for i, img in enumerate(final_images):
        print(f"Final Image {i+1}: bbox={img['bbox']}, source={img.get('source', 'unknown')}")
    
    # Save visualization
    output_path = f"outputs/comprehensive_deduplication_page_{page_num}.png"
    visualize_page(page_image, merged_results, output_path)
    print(f"\n✅ Visualization saved: {output_path}")
    
    print("\n=== DEDUPLICATION SUMMARY ===")
    total_before = len(image_elements) + len(vision_images)
    total_after = len(final_images)
    removed = total_before - total_after
    print(f"Image detections: {total_before} → {total_after} (removed: {removed})")
    
    if removed > 0:
        print("✅ Comprehensive deduplication successfully removed duplicate images!")
    else:
        print("ℹ️ No duplicate images found to remove.")
    
    print("\n✅ Comprehensive deduplication test completed!")

if __name__ == "__main__":
    test_comprehensive_deduplication()