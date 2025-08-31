import numpy as np
from shapely.geometry import box

def iou(box1, box2):
    b1 = box(*box1)
    b2 = box(*box2)
    inter = b1.intersection(b2).area
    union = b1.area + b2.area - inter
    return inter / union if union else 0

def merge_nearby_text_blocks(text_blocks, merge_threshold=10):
    """Merge text blocks that are close to each other"""
    if not text_blocks:
        return text_blocks
    
    merged = []
    used = set()
    
    for i, block1 in enumerate(text_blocks):
        if i in used:
            continue
            
        current_group = [block1]
        used.add(i)
        
        # Find nearby text blocks
        for j, block2 in enumerate(text_blocks):
            if j in used or i == j:
                continue
                
            # Check if blocks are close enough to merge
            if are_blocks_nearby(block1['bbox'], block2['bbox'], merge_threshold):
                current_group.append(block2)
                used.add(j)
        
        # Merge the group
        if len(current_group) == 1:
            merged.append(current_group[0])
        else:
            merged_block = merge_text_group(current_group)
            merged.append(merged_block)
    
    return merged

def are_blocks_nearby(bbox1, bbox2, threshold):
    """Check if two bounding boxes are close enough to merge"""
    # Calculate distances between boxes
    horizontal_gap = max(0, max(bbox1[0], bbox2[0]) - min(bbox1[2], bbox2[2]))
    vertical_gap = max(0, max(bbox1[1], bbox2[1]) - min(bbox1[3], bbox2[3]))
    
    return horizontal_gap <= threshold and vertical_gap <= threshold

def merge_text_group(text_group):
    """Merge a group of text blocks into one"""
    # Calculate bounding box that encompasses all blocks
    min_x = min(block['bbox'][0] for block in text_group)
    min_y = min(block['bbox'][1] for block in text_group)
    max_x = max(block['bbox'][2] for block in text_group)
    max_y = max(block['bbox'][3] for block in text_group)
    
    # Sort blocks by reading order (top to bottom, left to right)
    sorted_blocks = sorted(text_group, key=lambda b: (b['bbox'][1], b['bbox'][0]))
    
    # Combine text content
    combined_text = ' '.join(block.get('text', '') for block in sorted_blocks if block.get('text'))
    
    # Use the highest confidence score
    max_score = max(block.get('score', 0) for block in sorted_blocks)
    
    # Determine label based on content and formatting
    label = determine_text_label(combined_text, sorted_blocks)
    
    return {
        'label': label,
        'bbox': [min_x, min_y, max_x, max_y],
        'score': max_score,
        'text': combined_text
    }

def determine_text_label(text, blocks):
    """Determine the appropriate label for merged text"""
    if not text:
        return 'Text'
    
    # Check font size if available
    font_sizes = [block.get('font_size', 0) for block in blocks if block.get('font_size')]
    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
    
    # Title detection
    if avg_font_size > 20 or (text.isupper() and len(text.split()) <= 10):
        return 'Title'
    
    # Header detection
    if avg_font_size > 14 or (len(text.split()) <= 15 and any(word.isupper() for word in text.split())):
        return 'Header'
    
    # Default to text
    return 'Text'

def validate_table_detection(bbox, score, config=None):
    """Validate if a detected table is actually a valid table"""
    if not config:
        return True
    
    table_config = config.get('table_validation', {})
    
    # Calculate area
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    
    # Check minimum area
    min_area = table_config.get('min_area', 50000)
    if area < min_area:
        return False
    
    # Check aspect ratio
    if height > 0:
        aspect_ratio = width / height
        min_ratio = table_config.get('min_aspect_ratio', 0.3)
        max_ratio = table_config.get('max_aspect_ratio', 5.0)
        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
            return False
    
    return True

def remove_overlapping_tables(table_detections, config=None):
    """Remove overlapping table detections, keeping the one with highest confidence"""
    if not table_detections or not config:
        return table_detections
    
    table_config = config.get('table_validation', {})
    if not table_config.get('remove_overlapping', True):
        return table_detections
    
    overlap_threshold = table_config.get('overlap_threshold', 0.3)
    
    # Sort by confidence score (highest first)
    sorted_tables = sorted(table_detections, key=lambda x: x.get('score', 0), reverse=True)
    
    filtered_tables = []
    for table in sorted_tables:
        # Check if this table overlaps significantly with any already accepted table
        overlaps = False
        for accepted_table in filtered_tables:
            if iou(table['bbox'], accepted_table['bbox']) > overlap_threshold:
                overlaps = True
                break
        
        if not overlaps:
            filtered_tables.append(table)
    
    return filtered_tables

def merge_boxes(pdf_boxes, vision_boxes, iou_thresh=0.3, config=None):
    """
    Simplified approach: Prioritize native PDF text with strict table validation
    """
    merged = []
    
    # Configuration
    prioritize_native = config and config.get('text_detection', {}).get('prioritize_native_text', True)
    merge_nearby = config and config.get('text_detection', {}).get('merge_nearby_text', True)
    merge_threshold = config.get('text_detection', {}).get('text_merge_threshold', 5) if config else 5
    expand_pixels = config.get('text_detection', {}).get('expand_text_boxes', 3) if config else 3
    
    if prioritize_native:
        # Step 1: Process all PDF text elements first (they have accurate coordinates)
        text_elements = [p for p in pdf_boxes if p.get('type') == 'text' and p.get('text', '').strip()]
        
        # Merge nearby text blocks if enabled
        if merge_nearby:
            text_elements = merge_nearby_text_blocks(text_elements, merge_threshold)
        
        # Convert PDF text elements to final format with proper labeling
        for text_elem in text_elements:
            font_size = text_elem.get('font_size', 12)
            text = text_elem.get('text', '').strip()
            bbox = list(text_elem['bbox'])
            
            # Expand bounding box slightly for better visualization
            bbox[0] = max(0, bbox[0] - expand_pixels)  # left
            bbox[1] = max(0, bbox[1] - expand_pixels)  # top  
            bbox[2] = bbox[2] + expand_pixels  # right
            bbox[3] = bbox[3] + expand_pixels  # bottom
            
            # Classify text type based on content and font size
            if font_size > 18 or (len(text.split()) <= 8 and text.isupper()):
                label = 'Title'
            elif font_size > 14 or (len(text.split()) <= 12 and any(word.isupper() for word in text.split())):
                label = 'Header'  
            else:
                label = 'Text'
            
            merged.append({
                'label': label,
                'bbox': bbox,
                'score': 1.0,
                'text': text,
                'source': 'pdf_native'
            })
        
        # Step 2: Process and validate table detections
        table_detections = []
        other_detections = []
        
        for v_box in vision_boxes:
            # Skip text-like detections since we prioritize PDF text
            if any(text_type in v_box['label'].lower() for text_type in ['text', 'paragraph', 'title', 'header']):
                continue
            
            if 'table' in v_box['label'].lower():
                # Validate table detection
                if validate_table_detection(v_box['bbox'], v_box.get('score', 0), config):
                    table_detections.append({
                        'label': v_box['label'],
                        'bbox': v_box['bbox'],
                        'score': v_box.get('score', 0.5),
                        'source': 'vision'
                    })
            else:
                # Add other vision detections (images, etc.)
                other_detections.append({
                    'label': v_box['label'],
                    'bbox': v_box['bbox'],
                    'score': v_box.get('score', 0.5),
                    'source': 'vision'
                })
        
        # Remove overlapping table detections
        validated_tables = remove_overlapping_tables(table_detections, config)
        merged.extend(validated_tables)
        merged.extend(other_detections)
        
        # Step 3: Add non-text PDF elements
        for p in pdf_boxes:
            if p.get('type') == 'image':
                merged.append({
                    'label': 'Picture',
                    'bbox': p['bbox'],
                    'score': 1.0,
                    'source': 'pdf_native'
                })
    
    else:
        # Fallback to original merging approach
        merged = merge_boxes_original(pdf_boxes, vision_boxes, iou_thresh)
    
    return merged



def merge_boxes_original(pdf_boxes, vision_boxes, iou_thresh=0.3):
    """Original merging approach as fallback"""
    merged = []
    matched_pdf_ids = set()
    
    for v_box in vision_boxes:
        overlapping_pdfs = [p for p in pdf_boxes if iou(p['bbox'], v_box['bbox']) > iou_thresh]
        if overlapping_pdfs:
            min_x = min(v_box['bbox'][0], min(p['bbox'][0] for p in overlapping_pdfs))
            min_y = min(v_box['bbox'][1], min(p['bbox'][1] for p in overlapping_pdfs))
            max_x = max(v_box['bbox'][2], max(p['bbox'][2] for p in overlapping_pdfs))
            max_y = max(v_box['bbox'][3], max(p['bbox'][3] for p in overlapping_pdfs))
            merged_bbox = [min_x, min_y, max_x, max_y]
            merged_entry = {'label': v_box['label'], 'bbox': merged_bbox, 'score': v_box.get('score', 1.0)}
            
            texts = [p['text'] for p in overlapping_pdfs if 'text' in p and p['type'] == 'text']
            if texts:
                merged_entry['text'] = ' '.join(texts)
            
            merged.append(merged_entry)
            matched_pdf_ids.update(id(p) for p in overlapping_pdfs)
        else:
            merged.append(v_box)
    
    # Add unmatched PDF boxes
    for p in pdf_boxes:
        if id(p) not in matched_pdf_ids:
            if p['type'] == 'text':
                font_size = p.get('font_size', 0)
                if font_size > 20 or p['text'].isupper():
                    label = 'Title' if font_size > 20 else 'Header'
                else:
                    label = 'Text'
                entry = {'label': label, 'bbox': p['bbox'], 'score': 1.0, 'text': p['text']}
                merged.append(entry)
            elif p['type'] == 'image':
                entry = {'label': 'Picture', 'bbox': p['bbox'], 'score': 1.0}
                merged.append(entry)
    
    return merged

# Keep refine_graph as is
import networkx as nx

def refine_graph(boxes):
    G = nx.Graph()
    for i, b in enumerate(boxes):
        G.add_node(i, attr=b)
    for i in range(len(boxes)):
        for j in range(i+1, len(boxes)):
            if iou(boxes[i]['bbox'], boxes[j]['bbox']) > 0.3:
                G.add_edge(i, j)
    for edge in G.edges():
        n1, n2 = G.nodes[edge[0]]['attr'], G.nodes[edge[1]]['attr']
        if n1['label'] == 'caption' and n2['label'] == 'table':
            if n1['score'] < n2['score']:
                boxes[edge[0]]['label'] = 'paragraph'
    return boxes