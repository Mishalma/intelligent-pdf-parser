#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from parsers.pdf_parser import extract_text_with_pymupdf, render_page_to_image
from detectors.vision_detectors import VisionDetectors
from fusion.fusion import merge_boxes
from utils.output import save_visualization
import yaml

def test_text_removal():
    """Test text removal inside containers with different thresholds"""
    
    # Load configuration
    with open('src/configs/models.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    print("======================================================================")
    print("TESTING TEXT REMOVAL INSIDE CONTAINERS")
    print("======================================================================")
    
    pdf_path = "sample.pdf"
    page_num = 1  # Page with table
    
    # Initialize components
    vision_detectors = VisionDetectors(config)
    
    # Parse PDF
    print(f"Testing text removal on {pdf_path}, page {page_num}")
    pdf_boxes = extract_text_with_pymupdf(pdf_path, page_num, dpi=config.get('render_dpi', 300))
    
    # Get page image
    page_image, _ = render_page_to_image(pdf_path, page_num, dpi=config.get('render_dpi', 300))
    
    print(f"✅ PDF Analysis:")
    text_elements = [p for p in pdf_boxes if p.get('type') == 'text']
    print(f"   Text elements: {len(text_elements)}")
    
    # Vision detection
    print("✅ Running vision detection...")
    vision_boxes = vision_detectors.detect_layout(page_image)
    
    print(f"✅ Vision detection results:")
    table_detections = [v for v in vision_boxes if 'table' in v['label'].lower()]
    print(f"   Table detections: {len(table_detections)}")
    
    # Test different removal strategies
    print("\n=== TESTING DIFFERENT TEXT REMOVAL STRATEGIES ===")
    
    # Strategy 1: Current aggressive removal (50% threshold)
    print("\n1. Current aggressive removal (50% threshold):")
    merged_current = merge_boxes(pdf_boxes, vision_boxes, config=config)
    text_after_current = [elem for elem in merged_current if elem['label'].lower() in ['text', 'title', 'header']]
    print(f"   Text elements after removal: {len(text_after_current)}")
    
    # Save visualization
    save_visualization(merged_current, page_image, f"outputs/text_removal_current_page_{page_num}.png")
    print(f"   Visualization saved: outputs/text_removal_current_page_{page_num}.png")
    
    # Strategy 2: Very aggressive removal (30% threshold)
    print("\n2. Very aggressive removal (30% threshold):")
    from fusion.fusion import remove_contained_text_boxes_very_aggressive
    
    # Create a custom version with lower threshold
    def test_very_aggressive_removal(all_elements):
        text_elements = []
        container_elements = []
        
        for elem in all_elements:
            if elem['label'].lower() in ['text', 'title', 'header']:
                text_elements.append(elem)
            else:
                container_elements.append(elem)
        
        # Remove text that overlaps with containers (30% threshold)
        filtered_text_elements = []
        for text_elem in text_elements:
            is_contained = False
            for container_elem in container_elements:
                from fusion.fusion import is_text_inside_container
                if is_text_inside_container(text_elem['bbox'], container_elem['bbox'], threshold=0.3):
                    is_contained = True
                    print(f"   Removing text '{text_elem.get('text', '')[:30]}...' (30% overlap with {container_elem['label']})")
                    break
            
            if not is_contained:
                filtered_text_elements.append(text_elem)
        
        return filtered_text_elements + container_elements
    
    # Apply very aggressive removal
    merged_very_aggressive = merge_boxes(pdf_boxes, vision_boxes, config=config)
    merged_very_aggressive = test_very_aggressive_removal(merged_very_aggressive)
    text_after_very_aggressive = [elem for elem in merged_very_aggressive if elem['label'].lower() in ['text', 'title', 'header']]
    print(f"   Text elements after very aggressive removal: {len(text_after_very_aggressive)}")
    
    # Save visualization
    save_visualization(merged_very_aggressive, page_image, f"outputs/text_removal_very_aggressive_page_{page_num}.png")
    print(f"   Visualization saved: outputs/text_removal_very_aggressive_page_{page_num}.png")
    
    # Strategy 3: Ultra aggressive removal (20% threshold)
    print("\n3. Ultra aggressive removal (20% threshold):")
    
    def test_ultra_aggressive_removal(all_elements):
        text_elements = []
        container_elements = []
        
        for elem in all_elements:
            if elem['label'].lower() in ['text', 'title', 'header']:
                text_elements.append(elem)
            else:
                container_elements.append(elem)
        
        # Remove text that overlaps with containers (20% threshold)
        filtered_text_elements = []
        for text_elem in text_elements:
            is_contained = False
            for container_elem in container_elements:
                from fusion.fusion import is_text_inside_container
                if is_text_inside_container(text_elem['bbox'], container_elem['bbox'], threshold=0.2):
                    is_contained = True
                    print(f"   Removing text '{text_elem.get('text', '')[:30]}...' (20% overlap with {container_elem['label']})")
                    break
            
            if not is_contained:
                filtered_text_elements.append(text_elem)
        
        return filtered_text_elements + container_elements
    
    # Apply ultra aggressive removal
    merged_ultra_aggressive = merge_boxes(pdf_boxes, vision_boxes, config=config)
    merged_ultra_aggressive = test_ultra_aggressive_removal(merged_ultra_aggressive)
    text_after_ultra_aggressive = [elem for elem in merged_ultra_aggressive if elem['label'].lower() in ['text', 'title', 'header']]
    print(f"   Text elements after ultra aggressive removal: {len(text_after_ultra_aggressive)}")
    
    # Save visualization
    save_visualization(merged_ultra_aggressive, page_image, f"outputs/text_removal_ultra_aggressive_page_{page_num}.png")
    print(f"   Visualization saved: outputs/text_removal_ultra_aggressive_page_{page_num}.png")
    
    print("\n=== SUMMARY ===")
    print(f"Original text elements: {len(text_elements)}")
    print(f"After current removal (50%): {len(text_after_current)}")
    print(f"After very aggressive removal (30%): {len(text_after_very_aggressive)}")
    print(f"After ultra aggressive removal (20%): {len(text_after_ultra_aggressive)}")
    
    print("\n✅ Text removal test completed!")
    print("Check the generated visualizations to see the differences.")

if __name__ == "__main__":
    test_text_removal()