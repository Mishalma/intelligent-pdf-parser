#!/usr/bin/env python3
"""
Test script to demonstrate text filtering inside containers
"""

import sys
import os
sys.path.append('src')

from parsers.pdf_parser import render_page_to_image, parse_pdf_native
from detectors.vision_detectors import load_table_detector, detect_blocks
from fusion.fusion import merge_boxes
from utils.output import visualize_page
import yaml

def test_text_filtering(pdf_path="sample.pdf", page_num=0):
    """Test text filtering with a lower table threshold to show the effect"""
    
    # Load config and temporarily lower table threshold
    with open('src/configs/models.yaml') as f:
        config = yaml.safe_load(f)
    
    # Temporarily lower table threshold to detect some tables for demonstration
    original_threshold = config['table_detector']['confidence_threshold']
    config['table_detector']['confidence_threshold'] = 0.3  # Lower threshold
    config['table_validation']['min_area'] = 10000  # Lower area requirement
    
    print(f"Testing text filtering with lower table threshold: {config['table_detector']['confidence_threshold']}")
    
    try:
        # Step 1: Render page
        dpi = config['render_dpi']
        image, dims = render_page_to_image(pdf_path, page_num, dpi)
        print(f"✅ Rendered page: {dims[0]}x{dims[1]} pixels")
        
        # Step 2: Extract PDF elements
        pdf_elements = parse_pdf_native(pdf_path, page_num, dpi)
        text_elements = [e for e in pdf_elements if e.get('type') == 'text']
        print(f"✅ Extracted {len(text_elements)} text elements from PDF")
        
        # Step 3: Detect tables with lower threshold
        table_proc, table_model = load_table_detector(config['table_detector']['model_name'])
        vision_tables = detect_blocks(image, table_proc, table_model, config['table_detector']['confidence_threshold'])
        
        print(f"✅ Vision detected {len(vision_tables)} table candidates")
        for i, table in enumerate(vision_tables):
            bbox = table['bbox']
            score = table.get('score', 0)
            print(f"   Table {i+1}: bbox=({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}) score={score:.3f}")
        
        # Step 4: Test fusion with text filtering
        print(f"\n=== BEFORE TEXT FILTERING ===")
        print(f"Text elements: {len(text_elements)}")
        print(f"Table candidates: {len(vision_tables)}")
        
        # Merge with text filtering
        merged = merge_boxes(pdf_elements, vision_tables, config['iou_threshold'], config)
        
        final_text = [m for m in merged if m['label'].lower() in ['text', 'title', 'header']]
        final_tables = [m for m in merged if 'table' in m['label'].lower()]
        
        print(f"\n=== AFTER TEXT FILTERING ===")
        print(f"Final text elements: {len(final_text)}")
        print(f"Final table elements: {len(final_tables)}")
        
        # Step 5: Save visualization
        os.makedirs('outputs', exist_ok=True)
        output_path = f'outputs/text_filtering_demo_page_{page_num}.png'
        visualize_page(image, merged, output_path)
        print(f"✅ Visualization saved to {output_path}")
        
        # Show which text was filtered
        if len(final_text) < len(text_elements):
            filtered_count = len(text_elements) - len(final_text)
            print(f"✅ Successfully filtered {filtered_count} text boxes that were inside containers")
        else:
            print(f"ℹ️  No text boxes were filtered (no containers detected)")
        
        return merged
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with available PDF files
    test_files = ["sample.pdf", "resume.pdf", "samplenew.pdf"]
    
    for pdf_file in test_files:
        if os.path.exists(pdf_file):
            print(f"\n{'='*60}")
            print(f"TESTING TEXT FILTERING: {pdf_file}")
            print(f"{'='*60}")
            result = test_text_filtering(pdf_file, 0)
            if result:
                print(f"✅ Text filtering test completed for {pdf_file}")
            else:
                print(f"❌ Test failed for {pdf_file}")
            break
    else:
        print("❌ No test PDF files found.")