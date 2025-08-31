from transformers import DetrImageProcessor, DetrForObjectDetection, TableTransformerForObjectDetection, DetrForSegmentation, LayoutLMv3Processor, LayoutLMv3ForTokenClassification
import torch
import numpy as np
from PIL import Image

def load_block_detector(model_name):
    try:
        if "layoutlmv3" in model_name.lower():
            processor = LayoutLMv3Processor.from_pretrained(model_name)
            model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
        else:
            processor = DetrImageProcessor.from_pretrained(model_name)
            if "layout-detection" in model_name:
                model = DetrForSegmentation.from_pretrained(model_name)
            else:
                model = DetrForObjectDetection.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        print(f"Error loading block detector model {model_name}: {e}")
        raise

def load_table_detector(model_name="microsoft/table-transformer-detection"):
    try:
        processor = DetrImageProcessor.from_pretrained(model_name)
        model = TableTransformerForObjectDetection.from_pretrained(model_name)
        return processor, model
    except Exception as e:
        print(f"Error loading table detector model {model_name}: {e}")
        raise

def detect_blocks(image, processor, model, threshold=0.7):
    try:
        inputs = processor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=threshold)[0]
        
        boxes = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            if score > threshold:
                label_name = model.config.id2label[label.item()]
                bbox = box.tolist()
                
                # Normalize label names for consistency
                normalized_label = normalize_label(label_name)
                
                boxes.append({
                    'label': normalized_label,
                    'bbox': bbox,
                    'score': score.item()
                })
        return boxes
    except Exception as e:
        print(f"Error in block detection: {e}")
        raise

def normalize_label(label):
    """Normalize label names for consistency across models"""
    label_lower = label.lower()
    
    # Map various text labels to standard names
    if any(text_type in label_lower for text_type in ['text', 'paragraph', 'body']):
        return 'Text'
    elif any(title_type in label_lower for title_type in ['title', 'heading', 'header']):
        return 'Title'
    elif any(table_type in label_lower for table_type in ['table']):
        return 'Table'
    elif any(figure_type in label_lower for figure_type in ['figure', 'image', 'picture']):
        return 'Picture'
    elif any(caption_type in label_lower for caption_type in ['caption']):
        return 'Caption'
    else:
        return label  # Keep original if no mapping found

def detect_blocks_layoutlmv3(image, processor, model, threshold=0.7):
    """Specialized detection for LayoutLMv3 models"""
    try:
        # Convert PIL image to format expected by LayoutLMv3
        inputs = processor(image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Process LayoutLMv3 outputs (this is a simplified version)
        # In practice, you'd need to implement proper token-to-bbox mapping
        boxes = []
        # This is a placeholder - actual implementation would require 
        # proper handling of LayoutLMv3's token classification output
        return boxes
    except Exception as e:
        print(f"Error in LayoutLMv3 detection: {e}")
        return []

def ensemble_detect_blocks(image, models_and_processors, threshold=0.7):
    """Run multiple models and combine results"""
    all_boxes = []
    
    for processor, model in models_and_processors:
        try:
            boxes = detect_blocks(image, processor, model, threshold)
            all_boxes.extend(boxes)
        except Exception as e:
            print(f"Error in ensemble detection: {e}")
            continue
    
    # Remove duplicate detections using NMS-like approach
    return non_max_suppression(all_boxes, iou_threshold=0.3)

def non_max_suppression(boxes, iou_threshold=0.5):
    """Remove overlapping boxes with lower confidence"""
    if not boxes:
        return boxes
    
    # Sort by confidence score
    boxes = sorted(boxes, key=lambda x: x['score'], reverse=True)
    
    keep = []
    while boxes:
        current = boxes.pop(0)
        keep.append(current)
        
        # Remove boxes with high IoU
        boxes = [box for box in boxes if calculate_iou(current['bbox'], box['bbox']) < iou_threshold]
    
    return keep

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def detect_tables_by_structure(pdf_elements, image_dims, config=None):
    """Detect tables based on PDF structure analysis (lines, text alignment)"""
    if not config or not config.get('table_validation', {}).get('structure_analysis', False):
        return []
    
    tables = []
    text_elements = [e for e in pdf_elements if e.get('type') == 'text']
    line_elements = [e for e in pdf_elements if e.get('type') == 'line']
    
    # Group text elements by vertical alignment (potential table rows)
    row_groups = group_text_by_rows(text_elements)
    
    # Look for table patterns: multiple aligned rows with consistent spacing
    for i, row_group in enumerate(row_groups):
        if len(row_group) >= 3:  # At least 3 elements in a row
            # Check if this looks like a table row
            if is_table_row_pattern(row_group):
                # Look for adjacent rows that form a table
                table_rows = [row_group]
                
                # Check subsequent rows
                for j in range(i + 1, min(i + 10, len(row_groups))):  # Check up to 10 rows ahead
                    next_row = row_groups[j]
                    if len(next_row) >= 2 and is_aligned_with_table(row_group, next_row):
                        table_rows.append(next_row)
                    else:
                        break
                
                # If we found multiple aligned rows, create a table detection
                if len(table_rows) >= 2:
                    table_bbox = calculate_table_bbox(table_rows)
                    if validate_table_bbox(table_bbox, image_dims):
                        tables.append({
                            'label': 'Table',
                            'bbox': table_bbox,
                            'score': 0.8,  # High confidence for structure-based detection
                            'source': 'structure_analysis',
                            'rows': len(table_rows),
                            'columns': estimate_column_count(table_rows)
                        })
    
    return tables

def group_text_by_rows(text_elements, row_tolerance=10):
    """Group text elements into rows based on vertical alignment"""
    if not text_elements:
        return []
    
    # Sort by vertical position
    sorted_elements = sorted(text_elements, key=lambda e: e['bbox'][1])
    
    rows = []
    current_row = [sorted_elements[0]]
    current_y = sorted_elements[0]['bbox'][1]
    
    for element in sorted_elements[1:]:
        element_y = element['bbox'][1]
        
        # If element is close to current row, add it to the row
        if abs(element_y - current_y) <= row_tolerance:
            current_row.append(element)
        else:
            # Start new row
            if len(current_row) > 1:  # Only keep rows with multiple elements
                # Sort row elements by horizontal position
                current_row.sort(key=lambda e: e['bbox'][0])
                rows.append(current_row)
            current_row = [element]
            current_y = element_y
    
    # Add the last row
    if len(current_row) > 1:
        current_row.sort(key=lambda e: e['bbox'][0])
        rows.append(current_row)
    
    return rows

def is_table_row_pattern(row_elements):
    """Check if a row of text elements looks like a table row"""
    if len(row_elements) < 3:
        return False
    
    # Check for consistent spacing between elements
    spacings = []
    for i in range(len(row_elements) - 1):
        spacing = row_elements[i + 1]['bbox'][0] - row_elements[i]['bbox'][2]
        spacings.append(spacing)
    
    # Table rows typically have consistent or systematic spacing
    if len(spacings) >= 2:
        avg_spacing = sum(spacings) / len(spacings)
        # Check if spacings are reasonably consistent (within 50% of average)
        consistent_spacings = sum(1 for s in spacings if abs(s - avg_spacing) <= avg_spacing * 0.5)
        return consistent_spacings >= len(spacings) * 0.6  # At least 60% consistent
    
    return False

def is_aligned_with_table(reference_row, candidate_row):
    """Check if a candidate row aligns with a reference table row"""
    if len(candidate_row) < 2:
        return False
    
    # Check if columns align (within tolerance)
    ref_positions = [e['bbox'][0] for e in reference_row]
    candidate_positions = [e['bbox'][0] for e in candidate_row]
    
    alignment_tolerance = 20  # pixels
    aligned_columns = 0
    
    for cand_pos in candidate_positions:
        for ref_pos in ref_positions:
            if abs(cand_pos - ref_pos) <= alignment_tolerance:
                aligned_columns += 1
                break
    
    # Require at least 50% column alignment
    return aligned_columns >= min(len(candidate_positions), len(ref_positions)) * 0.5

def calculate_table_bbox(table_rows):
    """Calculate bounding box that encompasses all table rows"""
    all_elements = [elem for row in table_rows for elem in row]
    
    min_x = min(e['bbox'][0] for e in all_elements)
    min_y = min(e['bbox'][1] for e in all_elements)
    max_x = max(e['bbox'][2] for e in all_elements)
    max_y = max(e['bbox'][3] for e in all_elements)
    
    # Add some padding
    padding = 10
    return [min_x - padding, min_y - padding, max_x + padding, max_y + padding]

def validate_table_bbox(bbox, image_dims):
    """Validate that table bbox is reasonable"""
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    
    # Check minimum size
    if width < 100 or height < 50:
        return False
    
    # Check that bbox is within image bounds
    if bbox[0] < 0 or bbox[1] < 0 or bbox[2] > image_dims[0] or bbox[3] > image_dims[1]:
        return False
    
    return True

def estimate_column_count(table_rows):
    """Estimate the number of columns in the table"""
    if not table_rows:
        return 0
    
    # Use the row with the most elements as column count estimate
    return max(len(row) for row in table_rows)