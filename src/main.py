import yaml
import fitz
from parsers.pdf_parser import render_page_to_image, parse_pdf_native
from detectors.vision_detectors import load_block_detector, load_table_detector, detect_blocks
from fusion.cross_page import detect_headers_footers
from fusion.caption_linker import link_captions
from fusion.fusion import merge_boxes, refine_graph
from utils.output import save_json, visualize_page

# Load config
with open('src/configs/models.yaml') as f:
    config = yaml.safe_load(f)

def process_pdf(pdf_path):
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        all_pages_elements = []
        per_page_results = []
        
        # Load models
        models_and_processors = []
        
        # Load primary block detector
        block_proc, block_model = load_block_detector(config['block_detector']['model_name'])
        models_and_processors.append((block_proc, block_model))
        
        # Load ensemble models if configured
        if config.get('use_ensemble', False):
            for model_name in config.get('ensemble_models', []):
                if model_name != config['block_detector']['model_name']:
                    try:
                        proc, model = load_block_detector(model_name)
                        models_and_processors.append((proc, model))
                    except Exception as e:
                        print(f"Warning: Could not load ensemble model {model_name}: {e}")
        
        # Load table detector
        table_proc, table_model = load_table_detector(config['table_detector']['model_name'])
        
        # Per-page processing
        for page_num in range(num_pages):
            print(f"Processing page {page_num + 1}/{num_pages}")
            dpi = config['render_dpi']
            image, dims = render_page_to_image(pdf_path, page_num, dpi)
            pdf_elements = parse_pdf_native(pdf_path, page_num, dpi)  # Pass DPI for coordinate scaling
            all_pages_elements.append(pdf_elements)
            
            print(f"   Found {len([e for e in pdf_elements if e['type'] == 'text'])} text elements")
            
            # Vision detections
            block_boxes = detect_blocks(image, block_proc, block_model, config['block_detector']['confidence_threshold'])
            table_boxes = detect_blocks(image, table_proc, table_model, config['table_detector']['confidence_threshold'])
            
            # Initial merge
            vision_boxes = block_boxes + table_boxes
            merged = merge_boxes(pdf_elements, vision_boxes, config['iou_threshold'], config)
        
            per_page_results.append(merged)
        
        # Cross-page header/footer detection
        hf = detect_headers_footers(all_pages_elements)
        for hf_item in hf:
            per_page_results[hf_item['page']].append({
                'label': hf_item['label'],
                'bbox': hf_item['bbox'],
                'score': 1.0
            })
        
        # Caption linking and refinement
        for page_res in per_page_results:
            captions = [b for b in page_res if b['label'] == 'caption']
            targets = [b for b in page_res if b['label'] in ['image', 'table']]
            links = link_captions(captions, targets, config['caption_window'])
            # Optionally store links in results
            page_res = refine_graph(page_res)
            page_res.sort(key=lambda b: (b['bbox'][1], b['bbox'][0]))  # Reading order
        
        # Save outputs
        save_json(per_page_results, 'outputs/results.json')
        for page_num, res in enumerate(per_page_results):
            image, _ = render_page_to_image(pdf_path, page_num)
            visualize_page(image, res, f'outputs/page_{page_num}.png')
        
        print("Processing complete. Results saved in 'outputs/'.")
    
    except Exception as e:
        print(f"Error processing PDF: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', required=True)
    args = parser.parse_args()
    process_pdf(args.pdf)