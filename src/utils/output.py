import json
import cv2
import os
import numpy as np

def save_json(per_page_results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(per_page_results, f, indent=4)

def visualize_page(image, boxes, output_png):
    """Enhanced visualization with better colors and source indicators"""
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Enhanced color scheme based on element type and source
    for b in boxes:
        x0, y0, x1, y1 = map(int, b['bbox'])
        
        # Simple 3-color scheme: Red=Text, Blue=Image, Green=Table
        if 'table' in b['label'].lower():
            color = (0, 255, 0)  # Green for tables
            thickness = 3
        elif b['label'].lower() in ['picture', 'image', 'figure']:
            color = (255, 0, 0)  # Blue for images
            thickness = 2
        else:  # All text elements (Text, Title, Header)
            color = (0, 0, 255)  # Red for all text types
            thickness = 2
        
        # Adjust color intensity based on source
        source = b.get('source', 'unknown')
        if source == 'pdf_native':
            # Keep full intensity for native PDF elements
            pass
        elif source == 'structure_analysis':
            # Slightly darker for structure-based detections
            color = tuple(int(c * 0.8) for c in color)
        elif source == 'vision':
            # Even darker for vision-based detections
            color = tuple(int(c * 0.6) for c in color)
        
        # Draw rectangle
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), color, thickness)
        
        # Prepare label text
        display_label = b['label']
        source_indicator = ""
        
        if source == 'pdf_native':
            source_indicator = "📄"
        elif source == 'structure_analysis':
            source_indicator = "🔍"
        elif source == 'vision':
            source_indicator = "👁️"
        
        # Add text content for text elements
        if 'text' in b and b['text'] and b['label'].lower() in ['text', 'title', 'header']:
            shortened_text = b['text'].strip()[:25].replace('\n', ' ')
            if len(b['text']) > 25:
                shortened_text += "..."
            display_label = f"{shortened_text}"
        
        # Add additional info for tables
        if 'table' in b['label'].lower():
            rows = b.get('rows', '?')
            cols = b.get('columns', '?')
            display_label = f"Table {rows}x{cols}"
        
        # Create label with source and confidence
        label_text = f"{source_indicator}{display_label} ({b['score']:.2f})"
        
        # Calculate text position (avoid overlap)
        text_y = max(y0 - 10, 15)
        
        # Add text background for better readability
        text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
        cv2.rectangle(img_cv, (x0, text_y - text_size[1] - 5), 
                     (x0 + text_size[0] + 5, text_y + 5), (255, 255, 255), -1)
        
        # Add text
        cv2.putText(img_cv, label_text, (x0 + 2, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Add legend
    add_legend(img_cv)
    
    cv2.imwrite(output_png, img_cv)

def add_legend(img_cv):
    """Add a legend to explain the visualization"""
    height, width = img_cv.shape[:2]
    
    # Legend background
    legend_height = 120
    legend_width = 300
    legend_x = width - legend_width - 10
    legend_y = 10
    
    cv2.rectangle(img_cv, (legend_x, legend_y), 
                 (legend_x + legend_width, legend_y + legend_height), 
                 (255, 255, 255), -1)
    cv2.rectangle(img_cv, (legend_x, legend_y), 
                 (legend_x + legend_width, legend_y + legend_height), 
                 (0, 0, 0), 2)
    
    # Legend items
    legend_items = [
        ("Red = Text (all types)", (0, 0, 255)),
        ("Green = Tables", (0, 255, 0)),
        ("Blue = Images/Figures", (255, 0, 0)),
        ("📄=PDF 🔍=Structure 👁️=Vision", (0, 0, 0))
    ]
    
    for i, (text, color) in enumerate(legend_items):
        y_pos = legend_y + 20 + i * 20
        if i < 3:  # Draw colored rectangle for source types
            cv2.rectangle(img_cv, (legend_x + 10, y_pos - 8), 
                         (legend_x + 25, y_pos + 2), color, -1)
        cv2.putText(img_cv, text, (legend_x + 35 if i < 3 else legend_x + 10, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)