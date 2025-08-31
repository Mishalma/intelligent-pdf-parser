#!/usr/bin/env python3
"""
Test script for specific PDF pages
"""

import sys
import os
sys.path.append('src')

from parsers.pdf_parser import render_page_to_image, parse_pdf_native
from detectors.vision_detectors import load_block_detector, load_table_detector, detect_blocks
from fusion.fusion import merge_boxes
from utils.output import visualize_page
import yaml

def test_specific_page(pdf_path, page_num):
    """Test detection on a specific page"""
    
    # Load config
    with open('src/configs/models.yaml') as f:
        config = yaml.safe_load(f)
    
    print(f"Testing detection on {pdf_path}, page {page_num}")
    
    try:
        # Step 1: Render page
        dpi = config['render_dpi']
        image, dims = render_page_to_image(pdf_path, page_num, dpi)
        print(f"✅ Rendered page: {dims[0]}x{dims[1]} pixels")
        
        # Step 2: Extract PDF elements
        pdf_elements = parse_pdf_native(pdf_path, page_num, dpi)
        text_elements = [e for e in pdf_elements if e.get('type') == 'text']
        print(f"✅ Extracted {len(text_elements)} text elements from PDF")
        
        # Show sample text elements
        for i, elem in enumerate(text_elements[:5]):
            text = elem['text'][:50] + "..." if len(elem['text']) > 50 else elem['text']
            font_size = elem.get('font_size', 0)
            print(f"   Text {i+1}: '{text}' (font: {font_size:.1f})")
        
        # Step 3: Vision detection
        print(f"✅ Loading vision models...")
        
        # Load models
        block_proc, block_model = load_block_detector(config['block_detector']['model_name'])
        table_proc, table_model = load_table_detector(config['table_detector']['model_name'])
        
        # Run detection
        block_detections = detect_blocks(image, block_proc, block_model, config['block_detector']['confidence_threshold'])
        table_detections = detect_blocks(image, table_proc, table_model, config['table_detector']['confidence_threshold'])
        
        all_vision_detections = block_detections + table_detections
        
        print(f"✅ Vision detection results:")
        print(f"   Block detections: {len(block_detections)}")
        print(f"   Table detections: {len(table_detections)}")
        
        # Show vision detection breakdown
        vision_by_type = {}
        for det in all_vision_detections:
            label = det['label']
            vision_by_type[label] = vision_by_type.get(label, 0) + 1
        
        for label, count in vision_by_type.items():
            print(f"   {label}: {count}")
        
        # Step 4: Fusion
        merged = merge_boxes(pdf_elements, all_vision_detections, config['iou_threshold'], config)
        
        # Analyze final results
        final_by_type = {}
        for elem in merged:
            label = elem['label']
            final_by_type[label] = final_by_type.get(label, 0) + 1
        
        print(f"✅ Final merged results: {len(merged)} elements")
        for label, count in final_by_type.items():
            print(f"   {label}: {count}")
        
        # Step 5: Save visualization
        os.makedirs('outputs', exist_ok=True)
        output_path = f'outputs/page_{page_num}_detection.png'
        visualize_page(image, merged, output_path)
        print(f"✅ Visualization saved to {output_path}")
        
        return merged
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        pdf_file = sys.argv[1]
        page_num = int(sys.argv[2])
    else:
        # Test both PDF files on different pages
        test_cases = [
            ("sample.pdf", 0),
            ("samplenew.pdf", 0),
            ("sample.pdf", 1),  # Try page 1 if it exists
            ("samplenew.pdf", 1),  # Try page 1 if it exists
        ]
        
        for pdf_file, page_num in test_cases:
            if os.path.exists(pdf_file):
                print(f"\n{'='*60}")
                print(f"TESTING: {pdf_file} - Page {page_num}")
                print(f"{'='*60}")
                try:
                    result = test_specific_page(pdf_file, page_num)
                    if result:
                        print(f"✅ Test completed for {pdf_file} page {page_num}")
                    else:
                        print(f"❌ Test failed for {pdf_file} page {page_num}")
                except Exception as e:
                    print(f"❌ Could not test {pdf_file} page {page_num}: {e}")
        exit()
    
    # Test specific file and page
    if os.path.exists(pdf_file):
        result = test_specific_page(pdf_file, page_num)
        if result:
            print(f"✅ Test completed successfully")
        else:
            print(f"❌ Test failed")
    else:
        print(f"❌ File {pdf_file} not found")