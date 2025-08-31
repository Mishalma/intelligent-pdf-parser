#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

import yaml
from parsers.pdf_parser import render_page_to_image, extract_text_with_pymupdf
from detectors.vision_detectors import load_block_detector, load_table_detector, detect_blocks
from fusion.fusion import merge_boxes
from utils.output import visualize_page

def test_strict_validation():
    """Test the improved strict validation for tables and image deduplication"""
    
    # Load configuration
    with open('src/configs/models.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("======================================================================")
    print("TESTING STRICT TABLE & IMAGE VALIDATION")
    print("======================================================================")
    
    pdf_path = "sample.pdf"
    page_num = 1  # Test on page with table
    
    print(f"Testing strict validation on {pdf_path}, page {page_num}")
    print(f"Table confidence threshold: {config['table_detector']['confidence_threshold']}")
    print(f"Table min area: {config['table_validation']['min_area']}")
    print(f"Table min confidence: {config['table_validation']['min_confidence']}")
    print(f"Table min text elements: {config['table_validation']['min_text_elements']}")
    
    # Parse PDF
    pdf_boxes = extract_text_with_pymupdf(pdf_path, page_num, dpi=config.get('render_dpi', 300))
    page_image, _ = render_page_to_image(pdf_path, page_num, dpi=config.get('render_dpi', 300))
    
    print(f"✅ PDF Analysis:")
    text_elements = [p for p in pdf_boxes if p.get('type') == 'text']
    print(f"   Text elements: {len(text_elements)}")
    
    # Load models
    print("✅ Loading vision models...")
    block_proc, block_model = load_block_detector(config['block_detector']['model_name'])
    table_proc, table_model = load_table_detector(config['table_detector']['model_name'])
    
    # Vision detection
    print("✅ Running vision detection...")
    vision_boxes = detect_blocks(page_image, block_proc, block_model, 
                                threshold=config['block_detector']['confidence_threshold'])
    
    # Add table detections
    table_boxes = detect_blocks(page_image, table_proc, table_model,
                               threshold=config['table_detector']['confidence_threshold'])
    
    print(f"✅ Vision detection results:")
    block_detections = [v for v in vision_boxes if 'table' not in v['label'].lower()]
    table_detections = [v for v in table_boxes] + [v for v in vision_boxes if 'table' in v['label'].lower()]
    image_detections = [v for v in vision_boxes if v['label'].lower() in ['picture', 'image', 'figure']]
    
    print(f"   Block detections: {len(block_detections)}")
    print(f"   Table detections (before validation): {len(table_detections)}")
    print(f"   Image detections (before deduplication): {len(image_detections)}")
    
    # Combine all vision detections
    all_vision_boxes = vision_boxes + table_boxes
    
    # Merge with strict validation
    print("✅ Applying strict validation...")
    merged_results = merge_boxes(pdf_boxes, all_vision_boxes, config=config)
    
    # Analyze results
    final_tables = [elem for elem in merged_results if elem['label'].lower() == 'table']
    final_images = [elem for elem in merged_results if elem['label'].lower() in ['picture', 'image']]
    final_text = [elem for elem in merged_results if elem['label'].lower() in ['text', 'title', 'header']]
    
    print(f"✅ Final results after strict validation:")
    print(f"   Tables: {len(final_tables)}")
    print(f"   Images: {len(final_images)}")
    print(f"   Text elements: {len(final_text)}")
    print(f"   Total elements: {len(merged_results)}")
    
    # Save visualization
    output_path = f"outputs/strict_validation_page_{page_num}.png"
    visualize_page(page_image, merged_results, output_path)
    print(f"✅ Visualization saved: {output_path}")
    
    print("\n=== VALIDATION SUMMARY ===")
    print(f"Table detections: {len(table_detections)} → {len(final_tables)} (rejected: {len(table_detections) - len(final_tables)})")
    print(f"Image detections: {len(image_detections)} → {len(final_images)} (deduplicated: {len(image_detections) - len(final_images)})")
    
    print("\n✅ Strict validation test completed!")

if __name__ == "__main__":
    test_strict_validation()