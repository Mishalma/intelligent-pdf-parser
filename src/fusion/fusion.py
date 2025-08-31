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

def validate_table_detection(bbox, score, config=None, pdf_elements=None):
    """Enhanced validation for table detection with stricter criteria"""
    if not config:
        return True
    
    table_config = config.get('table_validation', {})
    
    # Calculate dimensions
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    
    # 1. Check minimum area (stricter)
    min_area = table_config.get('min_area', 75000)  # Increased from 50000
    if area < min_area:
        print(f"Rejecting table: area {area} < minimum {min_area}")
        return False
    
    # 2. Check aspect ratio (stricter)
    if height > 0:
        aspect_ratio = width / height
        min_ratio = table_config.get('min_aspect_ratio', 0.5)  # Increased from 0.4
        max_ratio = table_config.get('max_aspect_ratio', 3.5)  # Decreased from 4.0
        if aspect_ratio < min_ratio or aspect_ratio > max_ratio:
            print(f"Rejecting table: aspect ratio {aspect_ratio:.2f} outside range [{min_ratio}, {max_ratio}]")
            return False
    
    # 3. Check minimum confidence score (new)
    min_confidence = table_config.get('min_confidence', 0.85)  # High confidence required
    if score < min_confidence:
        print(f"Rejecting table: confidence {score:.3f} < minimum {min_confidence}")
        return False
    
    # 4. Check minimum dimensions (new)
    min_width = table_config.get('min_width', 200)
    min_height = table_config.get('min_height', 100)
    if width < min_width or height < min_height:
        print(f"Rejecting table: dimensions {width}x{height} too small (min: {min_width}x{min_height})")
        return False
    
    # 5. Content-based validation (new)
    if pdf_elements:
        # Count text elements inside the table bbox
        text_elements_inside = 0
        for elem in pdf_elements:
            if elem.get('type') == 'text':
                elem_bbox = elem['bbox']
                # Check if text element is mostly inside table bbox
                if (elem_bbox[0] >= bbox[0] and elem_bbox[1] >= bbox[1] and 
                    elem_bbox[2] <= bbox[2] and elem_bbox[3] <= bbox[3]):
                    text_elements_inside += 1
        
        # Require minimum number of text elements for a valid table
        min_text_elements = table_config.get('min_text_elements', 6)
        if text_elements_inside < min_text_elements:
            print(f"Rejecting table: only {text_elements_inside} text elements inside (min: {min_text_elements})")
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

def is_text_inside_container(text_bbox, container_bbox, threshold=0.8):
    """Check if a text box is mostly contained within another element (table, image, etc.)"""
    # Calculate intersection area
    x1 = max(text_bbox[0], container_bbox[0])
    y1 = max(text_bbox[1], container_bbox[1])
    x2 = min(text_bbox[2], container_bbox[2])
    y2 = min(text_bbox[3], container_bbox[3])
    
    if x2 <= x1 or y2 <= y1:
        return False  # No intersection
    
    intersection_area = (x2 - x1) * (y2 - y1)
    text_area = (text_bbox[2] - text_bbox[0]) * (text_bbox[3] - text_bbox[1])
    
    if text_area == 0:
        return False
    
    # If most of the text box is inside the container, consider it contained
    overlap_ratio = intersection_area / text_area
    return overlap_ratio > threshold

def remove_contained_text_boxes(all_elements):
    """Remove text boxes that are contained within tables, images, or other elements"""
    text_elements = []
    container_elements = []
    
    # Separate text elements from container elements (tables, images, etc.)
    for elem in all_elements:
        if elem['label'].lower() in ['text', 'title', 'header']:
            text_elements.append(elem)
        else:
            container_elements.append(elem)
    
    # Filter out text elements that are contained within containers
    filtered_text_elements = []
    for text_elem in text_elements:
        is_contained = False
        for container_elem in container_elements:
            if is_text_inside_container(text_elem['bbox'], container_elem['bbox']):
                is_contained = True
                print(f"Removing text '{text_elem.get('text', '')[:30]}...' contained in {container_elem['label']}")
                break
        
        if not is_contained:
            filtered_text_elements.append(text_elem)
    
    # Return filtered text elements + all container elements
    return filtered_text_elements + container_elements

def merge_boxes(pdf_boxes, vision_boxes, iou_thresh=0.3, config=None):
    """
    Simplified approach: Prioritize native PDF text with basic table validation
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
        
        # Step 2: Process vision detections
        table_detections = []
        other_detections = []
        
        for v_box in vision_boxes:
            # Skip text-like detections since we prioritize PDF text
            if any(text_type in v_box['label'].lower() for text_type in ['text', 'paragraph', 'title', 'header']):
                continue
            
            if 'table' in v_box['label'].lower():
                # Validate table detection with enhanced criteria
                if validate_table_detection(v_box['bbox'], v_box.get('score', 0), config, pdf_boxes):
                    table_detections.append({
                        'label': 'Table',
                        'bbox': v_box['bbox'],
                        'score': v_box.get('score', 0.5),
                        'source': 'vision'
                    })
                else:
                    print(f"Table detection rejected: bbox={v_box['bbox']}, score={v_box.get('score', 0):.3f}")
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
        
        # Step 3: Add non-text PDF elements first
        pdf_images = []
        for p in pdf_boxes:
            if p.get('type') == 'image':
                pdf_images.append({
                    'label': 'Picture',
                    'bbox': p['bbox'],
                    'score': 1.0,
                    'source': 'pdf_native'
                })
        
        # Step 4: Combine all image detections (vision + PDF native) and deduplicate
        all_image_detections = other_detections + pdf_images
        filtered_image_detections = remove_overlapping_images_comprehensive(all_image_detections)
        merged.extend(filtered_image_detections)
        
        # Step 5: Remove text boxes contained within tables/images (more aggressive)
        merged = remove_contained_text_boxes_aggressive(merged)
    
    else:
        # Fallback to original merging approach
        merged = merge_boxes_original(pdf_boxes, vision_boxes, iou_thresh)
    
    return merged

def remove_overlapping_images(image_detections):
    """Enhanced image deduplication with stricter overlap detection"""
    if len(image_detections) <= 1:
        return image_detections
    
    # Filter to only image-type detections
    images = [det for det in image_detections if det['label'].lower() in ['picture', 'image', 'figure']]
    non_images = [det for det in image_detections if det['label'].lower() not in ['picture', 'image', 'figure']]
    
    if len(images) <= 1:
        return image_detections
    
    # Sort by confidence score (highest first)
    sorted_images = sorted(images, key=lambda x: x.get('score', 0), reverse=True)
    
    filtered_images = []
    for image in sorted_images:
        # Check if this image overlaps significantly with any already accepted image
        overlaps = False
        for accepted_image in filtered_images:
            overlap_iou = iou(image['bbox'], accepted_image['bbox'])
            
            # More aggressive deduplication with multiple criteria
            if overlap_iou > 0.3:  # Lowered from 0.5 to 0.3 for better deduplication
                overlaps = True
                print(f"Removing overlapping image detection (IoU: {overlap_iou:.3f} > 0.3)")
                break
            
            # Additional check: if images are very close in position (even with low IoU)
            img_center_x = (image['bbox'][0] + image['bbox'][2]) / 2
            img_center_y = (image['bbox'][1] + image['bbox'][3]) / 2
            acc_center_x = (accepted_image['bbox'][0] + accepted_image['bbox'][2]) / 2
            acc_center_y = (accepted_image['bbox'][1] + accepted_image['bbox'][3]) / 2
            
            distance = ((img_center_x - acc_center_x) ** 2 + (img_center_y - acc_center_y) ** 2) ** 0.5
            
            # If centers are very close (within 50 pixels), consider it a duplicate
            if distance < 50:
                overlaps = True
                print(f"Removing nearby image detection (distance: {distance:.1f} < 50 pixels)")
                break
        
        if not overlaps:
            filtered_images.append(image)
    
    return filtered_images + non_images

def remove_overlapping_images_comprehensive(all_detections):
    """Comprehensive image deduplication across vision and PDF native detections"""
    if len(all_detections) <= 1:
        return all_detections
    
    # Filter to only image-type detections
    images = [det for det in all_detections if det['label'].lower() in ['picture', 'image', 'figure']]
    non_images = [det for det in all_detections if det['label'].lower() not in ['picture', 'image', 'figure']]
    
    if len(images) <= 1:
        return all_detections
    
    # Sort by priority: PDF native first (score 1.0), then by confidence score
    def sort_priority(det):
        if det.get('source') == 'pdf_native':
            return (1, det.get('score', 0))  # PDF native gets highest priority
        else:
            return (0, det.get('score', 0))  # Vision detections get lower priority
    
    sorted_images = sorted(images, key=sort_priority, reverse=True)
    
    filtered_images = []
    for image in sorted_images:
        # Check if this image overlaps significantly with any already accepted image
        overlaps = False
        for accepted_image in filtered_images:
            overlap_iou = iou(image['bbox'], accepted_image['bbox'])
            
            # Very aggressive deduplication for comprehensive removal
            if overlap_iou > 0.2:  # Even lower threshold for comprehensive deduplication
                overlaps = True
                print(f"Removing overlapping image detection (IoU: {overlap_iou:.3f} > 0.2, source: {image.get('source', 'unknown')})")
                break
            
            # Additional check: if images are very close in position
            img_center_x = (image['bbox'][0] + image['bbox'][2]) / 2
            img_center_y = (image['bbox'][1] + image['bbox'][3]) / 2
            acc_center_x = (accepted_image['bbox'][0] + accepted_image['bbox'][2]) / 2
            acc_center_y = (accepted_image['bbox'][1] + accepted_image['bbox'][3]) / 2
            
            distance = ((img_center_x - acc_center_x) ** 2 + (img_center_y - acc_center_y) ** 2) ** 0.5
            
            # If centers are very close (within 75 pixels), consider it a duplicate
            if distance < 75:
                overlaps = True
                print(f"Removing nearby image detection (distance: {distance:.1f} < 75 pixels, source: {image.get('source', 'unknown')})")
                break
            
            # Additional check: if bounding boxes have significant overlap in any dimension
            x_overlap = min(image['bbox'][2], accepted_image['bbox'][2]) - max(image['bbox'][0], accepted_image['bbox'][0])
            y_overlap = min(image['bbox'][3], accepted_image['bbox'][3]) - max(image['bbox'][1], accepted_image['bbox'][1])
            
            if x_overlap > 0 and y_overlap > 0:
                # Calculate overlap percentage relative to smaller box
                img_area = (image['bbox'][2] - image['bbox'][0]) * (image['bbox'][3] - image['bbox'][1])
                acc_area = (accepted_image['bbox'][2] - accepted_image['bbox'][0]) * (accepted_image['bbox'][3] - accepted_image['bbox'][1])
                overlap_area = x_overlap * y_overlap
                
                smaller_area = min(img_area, acc_area)
                if smaller_area > 0 and (overlap_area / smaller_area) > 0.3:
                    overlaps = True
                    print(f"Removing overlapping image detection (area overlap: {overlap_area/smaller_area:.3f} > 0.3, source: {image.get('source', 'unknown')})")
                    break
        
        if not overlaps:
            filtered_images.append(image)
    
    return filtered_images + non_images

def remove_contained_text_boxes_aggressive(all_elements):
    """Aggressively remove text boxes that are contained within tables, images, or other elements"""
    text_elements = []
    container_elements = []
    
    # Separate text elements from container elements (tables, images, etc.)
    for elem in all_elements:
        if elem['label'].lower() in ['text', 'title', 'header']:
            text_elements.append(elem)
        else:
            container_elements.append(elem)
    
    # Remove text that overlaps significantly with containers
    filtered_text_elements = []
    for text_elem in text_elements:
        is_contained = False
        for container_elem in container_elements:
            # Use very low threshold for ultra aggressive removal - any overlap removes the text
            if is_text_inside_container(text_elem['bbox'], container_elem['bbox'], threshold=0.1):  # 10% overlap
                is_contained = True
                print(f"Removing text '{text_elem.get('text', '')[:30]}...' contained in {container_elem['label']}")
                break
        
        if not is_contained:
            filtered_text_elements.append(text_elem)
    
    # Return filtered text elements + all container elements
    return filtered_text_elements + container_elements

def remove_contained_text_boxes_simple(all_elements):
    """Simplified version - only remove text clearly inside tables"""
    text_elements = []
    container_elements = []
    
    # Separate text elements from container elements (tables, images, etc.)
    for elem in all_elements:
        if elem['label'].lower() in ['text', 'title', 'header']:
            text_elements.append(elem)
        else:
            container_elements.append(elem)
    
    # Only remove text that is very clearly inside containers
    filtered_text_elements = []
    for text_elem in text_elements:
        is_contained = False
        for container_elem in container_elements:
            if is_text_inside_container(text_elem['bbox'], container_elem['bbox'], threshold=0.9):  # Higher threshold
                is_contained = True
                print(f"Removing text '{text_elem.get('text', '')[:30]}...' contained in {container_elem['label']}")
                break
        
        if not is_contained:
            filtered_text_elements.append(text_elem)
    
    # Return filtered text elements + all container elements
    return filtered_text_elements + container_elements



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