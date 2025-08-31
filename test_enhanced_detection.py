#!/usr/bin/env python3
"""
Test script for enhanced image, table, and text detection accuracy
"""

import sys
import os
sys.path.append('src')

from parsers.pdf_parser import render_page_to_image, parse_pdf_native
from detectors.vision_detectors import load_block_detector, load_table_detector, detect_blocks, detect_tables_by_structure
from fusion.fusion import merge_boxes
from utils.output import visualize_page
import yaml

def test_enhanced_detection(pdf_path="sample.pdf", page_num=0):
    """Test the enhanced detection system with all improvements"""
    
    # Load config
    with open('src/configs/models.yaml') as f:
        config = yaml.safe_load(f)
    
    print(f"Testing enhanced detection on {pdf_path}, page {page_num}")
    print(f"Block model: {config['block_detector']['model_name']}")
    print(f"Block confidence: {config['block_detector']['confidence_threshold']}")
    print(f"Table model: {config['table_detector']['model_name']}")
    print(f"Table confidence: {config['table_detector']['confidence_threshold']}")
    print(f"Structure analysis: {config.get('table_validation', {}).get('structure_analysis', False)}")
    
    try:
        # Step 1: Render page
        dpi = config['render_dpi']
        image, dims = render_page_to_image(pdf_path, page_num, dpi)
        print(f"✅ Rendered page: {dims[0]}x{dims[1]} pixels")
        
        # Step 2: Enhanced PDF parsing with structure extraction
        pdf_elements = parse_pdf_native(pdf_path, page_num, dpi)
        
        # Analyze extracted elements
        text_elements = [e for e in pdf_elements if e.get('type') == 'text']
        line_elements = [e for e in pdf_elements if e.get('type') == 'line']
        image_elements = [e for e in pdf_elements if e.get('type') == 'image']
        
        print(f"✅ Enhanced PDF Analysis:")
        print(f"   Text elements: {len(text_elements)}")
        print(f"   Line elements: {len(line_elements)}")
        print(f"   Image elements: {len(image_elements)}")
        
        # Show sample text with font sizes
        for i, elem in enumerate(text_elements[:3]):
            text = elem['text'][:40] + "..." if len(elem['text']) > 40 else elem['text']
            font_size = elem.get('font_size', 0)
            print(f"   Text {i+1}: '{text}' (font: {font_size:.1f})")
        
        # Step 3: Structure-based table detection
        structure_tables = detect_tables_by_structure(pdf_elements, dims, config)
        print(f"✅ Structure-based table detection: {len(structure_tables)} tables")
        
        for i, table in enumerate(structure_tables):
            bbox = table['bbox']
            rows = table.get('rows', 0)
            cols = table.get('columns', 0)
            score = table.get('score', 0)
            print(f"   Structure Table {i+1}: {rows}x{cols} at ({bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}) score={score:.2f}")
        
        # Step 4: Vision-based detection
        print(f"✅ Loading vision models...")
        
        # Load block detector
        block_proc, block_model = load_block_detector(config['block_detector']['model_name'])
        block_detections = detect_blocks(image, block_proc, block_model, config['block_detector']['confidence_threshold'])
        
        # Load table detector
        table_proc, table_model = load_table_detector(config['table_detector']['model_name'])
        table_detections = detect_blocks(image, table_proc, table_model, config['table_detector']['confidence_threshold'])
        
        # Combine vision detections
        all_vision_detections = block_detections + table_detections
        
        print(f"✅ Vision detection results:")
        print(f"   Block detections: {len(block_detections)}")
        print(f"   Table detections: {len(table_detections)}")
        print(f"   Total vision detections: {len(all_vision_detections)}")
        
        # Show vision detection breakdown by type
        vision_by_type = {}
        for det in all_vision_detections:
            label = det['label']
            vision_by_type[label] = vision_by_type.get(label, 0) + 1
        
        for label, count in vision_by_type.items():
            print(f"   {label}: {count}")
        
        # Step 5: Enhanced fusion with multi-source table detection
        merged = merge_boxes(pdf_elements, all_vision_detections, config['iou_threshold'], config)
        
        # Analyze final results
        final_by_type = {}
        for elem in merged:
            label = elem['label']
            final_by_type[label] = final_by_type.get(label, 0) + 1
        
        print(f"✅ Final merged results: {len(merged)} elements")
        for label, count in final_by_type.items():
            print(f"   {label}: {count}")
        
        # Step 6: Save enhanced visualization
        os.makedirs('outputs', exist_ok=True)
        output_path = f'outputs/enhanced_detection_page_{page_num}.png'
        visualize_page(image, merged, output_path)
        print(f"✅ Enhanced visualization saved to {output_path}")
        
        # Step 7: Detailed accuracy analysis
        print(f"\n=== ACCURACY ANALYSIS ===")
        
        # Text accuracy
        text_merged = [m for m in merged if m['label'] in ['Text', 'Title', 'Header']]
        print(f"Text detection: {len(text_elements)} native → {len(text_merged)} final")
        
        # Table accuracy
        table_merged = [m for m in merged if 'table' in m['label'].lower()]
        total_table_sources = len(structure_tables) + len(table_detections)
        print(f"Table detection: {total_table_sources} candidates → {len(table_merged)} final")
        
        # Image accuracy
        image_merged = [m for m in merged if m['label'] in ['Picture', 'Image', 'Figure']]
        print(f"Image detection: {len(image_elements)} native + vision → {len(image_merged)} final")
        
        # Coordinate validation
        valid_coords = 0
        for elem in merged:
            bbox = elem['bbox']
            if (0 <= bbox[0] < bbox[2] <= dims[0] and 
                0 <= bbox[1] < bbox[3] <= dims[1]):
                valid_coords += 1
        
        print(f"Coordinate accuracy: {valid_coords}/{len(merged)} ({100*valid_coords/max(len(merged),1):.1f}%)")
        
        return merged
        
    except Exception as e:
        print(f"❌ Error during enhanced detection test: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test with available PDF files
    test_files = ["sample.pdf", "resume.pdf", "samplenew.pdf"]
    
    for pdf_file in test_files:
        if os.path.exists(pdf_file):
            print(f"\n{'='*70}")
            print(f"TESTING ENHANCED DETECTION: {pdf_file}")
            print(f"{'='*70}")
            result = test_enhanced_detection(pdf_file, 0)
            if result:
                print(f"✅ Enhanced detection test completed successfully for {pdf_file}")
                print(f"   Total elements detected: {len(result)}")
            else:
                print(f"❌ Enhanced detection test failed for {pdf_file}")
            break
    else:
        print("❌ No test PDF files found. Please ensure sample.pdf, resume.pdf, or samplenew.pdf exists.")