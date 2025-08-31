#!/usr/bin/env python3
"""
Test script specifically for table detection accuracy
"""

import sys
import os
sys.path.append('src')

from parsers.pdf_parser import render_page_to_image, parse_pdf_native
from detectors.vision_detectors import load_table_detector, detect_blocks
from fusion.fusion import merge_boxes
from utils.output import visualize_page
import yaml

def test_table_detection(pdf_path="sample.pdf", page_num=0):
    """Test enhanced table detection"""
    
    # Load config
    with open('src/configs/models.yaml') as f:
        config = yaml.safe_load(f)
    
    print(f"Testing table detection on {pdf_path}, page {page_num}")
    print(f"Table model: {config['table_detector']['model_name']}")
    print(f"Table confidence threshold: {config['table_detector']['confidence_threshold']}")
    print(f"Structure analysis enabled: {config.get('table_detection', {}).get('use_structure_analysis', True)}")
    
    try:
        # Step 1: Render page
        dpi = config['render_dpi']
        image, dims = render_page_to_image(pdf_path, page_num, dpi)
        print(f"✅ Rendered page: {dims[0]}x{dims[1]} pixels")
        
        # Step 2: Extract PDF elements including table structures
        pdf_elements = parse_pdf_native(pdf_path, page_num, dpi)
        
        # Analyze extracted elements
        text_elements = [e for e in pdf_elements if e.get('type') == 'text']
        line_elements = [e for e in pdf_elements if e.get('type') == 'line']
        table_candidates = [e for e in pdf_elements if e.get('type') == 'table']
        
        print(f"✅ PDF Analysis:")
        print(f"   Text elements: {len(text_elements)}")
        print(f"   Line elements: {len(line_elements)}")
        print(f"   Table candidates from structure: {len(table_candidates)}")
        
        # Show table candidates
        for i, table in enumerate(table_candidates):
            bbox = table['bbox']
            h_lines = table.get('horizontal_lines', 0)
            v_lines = table.get('vertical_lines', 0)
            confidence = table.get('confidence', 0)
            print(f"   Table {i+1}: bbox=({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}) "
                  f"lines=({h_lines}H, {v_lines}V) conf={confidence:.2f}")
        
        # Step 3: Vision-based table detection
        table_proc, table_model = load_table_detector(config['table_detector']['model_name'])
        vision_tables = detect_blocks(image, table_proc, table_model, config['table_detector']['confidence_threshold'])
        
        print(f"✅ Vision table detection: {len(vision_tables)} tables found")
        for i, table in enumerate(vision_tables):
            bbox = table['bbox']
            score = table.get('score', 0)
            print(f"   Vision Table {i+1}: bbox=({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}) "
                  f"score={score:.3f}")
        
        # Step 4: Simple table processing (just use vision detections)
        print(f"✅ Using vision table detections directly")
        
        # Step 5: Full pipeline test
        all_vision_boxes = vision_tables  # For this test, focus on tables
        merged_all = merge_boxes(pdf_elements, all_vision_boxes, config['iou_threshold'], config)
        final_tables = [m for m in merged_all if 'table' in m['label'].lower()]
        
        print(f"✅ Full pipeline result: {len(final_tables)} tables in final output")
        
        # Step 6: Save visualization
        os.makedirs('outputs', exist_ok=True)
        output_path = f'outputs/table_detection_page_{page_num}.png'
        
        # Create visualization with all elements but highlight tables
        visualize_page(image, merged_all, output_path)
        print(f"✅ Visualization saved to {output_path}")
        
        # Summary
        print(f"\n=== TABLE DETECTION SUMMARY ===")
        print(f"PDF structure analysis: {len(table_candidates)} candidates")
        print(f"Vision model detection: {len(vision_tables)} tables")
        print(f"Final merged tables: {len(final_tables)} tables")
        
        if len(final_tables) > 0:
            print("✅ Table detection successful!")
        else:
            print("⚠️  No tables detected - may need parameter tuning")
        
        return merged_all
        
    except Exception as e:
        print(f"❌ Error during table detection test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with available PDF files
    test_files = ["sample.pdf", "resume.pdf", "samplenew.pdf"]
    
    for pdf_file in test_files:
        if os.path.exists(pdf_file):
            print(f"\n{'='*60}")
            print(f"TESTING TABLE DETECTION: {pdf_file}")
            print(f"{'='*60}")
            result = test_table_detection(pdf_file, 0)
            if result:
                print(f"✅ Table detection test completed for {pdf_file}")
            else:
                print(f"❌ Table detection test failed for {pdf_file}")
            break
    else:
        print("❌ No test PDF files found. Please ensure sample.pdf, resume.pdf, or samplenew.pdf exists.")