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
                
                boxes.append({
                    'label': label_name,
                    'bbox': bbox,
                    'score': score.item()
                })
        return boxes
    except Exception as e:
        print(f"Error in block detection: {e}")
        raise

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