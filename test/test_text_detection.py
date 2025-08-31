#!/usr/bin/env python3
"""
Test script to verify improved text detection accuracy
"""

import yaml
import sys
import os
sys.path.append('src')

from parsers.pdf_parser import render_page_to_image, parse_pdf_native
from detectors.vision_detectors import load_block_detector, detect_blocks
from fusion.fusion import merge_boxes
from utils.output import visualize_page

def test_text_detection(pdf_path="sample.pdf", page_num=0):
    """Test the improved text detection on a sample PDF"""
    
    # Load config
    with open('src/configs/models.yaml') as f:
        config = yaml.safe_load(f)
    
    print(f"Testing text detection on {pdf_path}, page {page_num}")
    print(f"Using model: {config['block_detector']['model_name']}")
    print(f"Confidence threshold: {config['block_detector']['confidence_threshold']}")
    print(f"Render DPI: {config['render_dpi']}")
    
    try:
        # Render page
        print("\n1. Rendering page to image...")
        image, dims = render_page_to_image(pdf_path, page_num, config['render_dpi'])
        print(f"   Image dimensions: {dims}")
        
        # Extract native PDF elements
        print("\n2. Extracting native PDF elements...")
        pdf_elements = parse_pdf_native(pdf_path, page_num)
        text_elements = [e for e in pdf_elements if e['type'] == 'text']
        print(f"   Found {len(text_elements)} text elements")
        for i, elem in enumerate(text_elements[:5]):  # Show first 5
            print(f"   Text {i+1}: '{elem['text'][:50]}...' (font: {elem.get('font_size', 'unknown')})")
        
        # Load vision model
        print("\n3. Loading vision detection model...")
        block_proc, block_model = load_block_detector(config['block_detector']['model_name'])
        
        # Run vision detection
        print("\n4. Running vision detection...")
        vision_boxes = detect_blocks(image, block_proc, block_model, config['block_detector']['confidence_threshold'])
        print(f"   Found {len(vision_boxes)} vision detections")
        for i, box in enumerate(vision_boxes[:5]):  # Show first 5
            print(f"   Detection {i+1}: {box['label']} (confidence: {box['score']:.3f})")
        
        # Merge results
        print("\n5. Merging results...")
        merged = merge_boxes(pdf_elements, vision_boxes, config['iou_threshold'], config)
        text_merged = [m for m in merged if 'text' in m['label'].lower() or m['label'] in ['Text', 'Title', 'Header']]
        print(f"   Final merged text elements: {len(text_merged)}")
        
        # Save visualization
        print("\n6. Saving visualization...")
        os.makedirs('outputs', exist_ok=True)
        visualize_page(image, merged, f'outputs/test_detection_page_{page_num}.png')
        print(f"   Visualization saved to outputs/test_detection_page_{page_num}.png")
        
        # Print summary
        print(f"\n=== SUMMARY ===")
        print(f"Native PDF text elements: {len(text_elements)}")
        print(f"Vision detections: {len(vision_boxes)}")
        print(f"Final merged text elements: {len(text_merged)}")
        print(f"Improvement ratio: {len(text_merged) / max(len(text_elements), 1):.2f}x")
        
        return merged
        
    except Exception as e:
        print(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with available PDF files
    test_files = ["sample.pdf", "resume.pdf", "samplenew.pdf"]
    
    for pdf_file in test_files:
        if os.path.exists(pdf_file):
            print(f"\n{'='*60}")
            print(f"TESTING: {pdf_file}")
            print(f"{'='*60}")
            result = test_text_detection(pdf_file, 0)
            if result:
                print(f"✅ Test completed successfully for {pdf_file}")
            else:
                print(f"❌ Test failed for {pdf_file}")
            break
    else:
        print("No test PDF files found. Please ensure sample.pdf, resume.pdf, or samplenew.pdf exists.")