import json
import cv2
import os
import numpy as np

def save_json(per_page_results, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(per_page_results, f, indent=4)

def visualize_page(image, boxes, output_png):
    os.makedirs(os.path.dirname(output_png), exist_ok=True)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    for b in boxes:
        x0, y0, x1, y1 = map(int, b['bbox'])
        if b['label'].lower() in ['table']:
            color = (0, 255, 0)  # Green
        elif b['label'].lower() in ['picture', 'image', 'figure']:
            color = (255, 0, 0)  # Blue
        else:  # Text, Title, Header, etc.
            color = (0, 0, 255)  # Red
        cv2.rectangle(img_cv, (x0, y0), (x1, y1), color, 2)
        display_label = b['label']
        if 'text' in b and b['text']:
            shortened_text = b['text'].strip()[:30].replace('\n', ' ') + '...' if len(b['text']) > 30 else b['text']
            display_label = f"Text: {shortened_text}"
        cv2.putText(img_cv, f"{display_label} ({b['score']:.2f})", (x0, y0 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    cv2.imwrite(output_png, img_cv)