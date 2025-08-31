#!/usr/bin/env python3
"""
Test the improved accurate text detection
"""

import sys
import os
sys.path.append('src')

from parsers.pdf_parser import render_page_to_image, parse_pdf_native
from fusion.fusion import merge_boxes
from utils.output import visualize_page
import yaml

def test_accurate_detection(pdf_path="sample.pdf", page_num=0):
    """Test the new accurate text detection approach"""
    
    # Load config
    with open('src/configs/models.yaml') as f:
        config = yaml.safe_load(f)
    
    print(f"Testing accurate text detection on {pdf_path}, page {page_num}")
    print(f"DPI: {config['render_dpi']}")
    print(f"Prioritize native text: {config.get('text_detection', {}).get('prioritize_native_text', True)}")
    
    try:
        # Step 1: Render page
        dpi = config['render_dpi']
        image, dims = render_page_to_image(pdf_path, page_num, dpi)
        print(f"✅ Rendered page: {dims[0]}x{dims[1]} pixels")
        
        # Step 2: Extract native PDF elements with proper coordinate scaling
        pdf_elements = parse_pdf_native(pdf_path, page_num, dpi)
        text_elements = [e for e in pdf_elements if e['type'] == 'text']
        print(f"✅ Extracted {len(text_elements)} text elements from PDF")
        
        # Show sample text elements with coordinates
        for i, elem in enumerate(text_elements[:5]):
            bbox = elem['bbox']
            text = elem['text'][:50] + "..." if len(elem['text']) > 50 else elem['text']
            font_size = elem.get('font_size', 0)
            print(f"   Text {i+1}: '{text}' at ({bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}) font:{font_size:.1f}")
        
        # Step 3: Use native-text-prioritized fusion (no vision model needed for this test)
        vision_boxes = []  # Empty for this test
        merged = merge_boxes(pdf_elements, vision_boxes, config['iou_threshold'], config)
        text_merged = [m for m in merged if m.get('text')]
        print(f"✅ Final merged text elements: {len(text_merged)}")
        
        # Step 4: Save visualization
        os.makedirs('outputs', exist_ok=True)
        output_path = f'outputs/accurate_detection_page_{page_num}.png'
        visualize_page(image, merged, output_path)
        print(f"✅ Visualization saved to {output_path}")
        
        # Step 5: Print summary with coordinate validation
        print(f"\n=== COORDINATE VALIDATION ===")
        img_width, img_height = dims
        valid_coords = 0
        for elem in text_merged:
            bbox = elem['bbox']
            if (0 <= bbox[0] < bbox[2] <= img_width and 
                0 <= bbox[1] < bbox[3] <= img_height):
                valid_coords += 1
            else:
                print(f"⚠️  Invalid coordinates: {bbox} (image: {img_width}x{img_height})")
        
        print(f"Valid coordinates: {valid_coords}/{len(text_merged)} ({100*valid_coords/max(len(text_merged),1):.1f}%)")
        
        return merged
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with available PDF files
    test_files = ["sample.pdf", "resume.pdf", "samplenew.pdf"]
    
    for pdf_file in test_files:
        if os.path.exists(pdf_file):
            print(f"\n{'='*60}")
            print(f"TESTING ACCURATE DETECTION: {pdf_file}")
            print(f"{'='*60}")
            result = test_accurate_detection(pdf_file, 0)
            if result:
                print(f"✅ Accurate detection test completed successfully for {pdf_file}")
            else:
                print(f"❌ Test failed for {pdf_file}")
            break
    else:
        print("❌ No test PDF files found. Please ensure sample.pdf, resume.pdf, or samplenew.pdf exists.")