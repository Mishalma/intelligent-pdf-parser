import fitz  # PyMuPDF
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTImage, LTLine, LTTextLineHorizontal, LTChar
import io
from PIL import Image

def render_page_to_image(pdf_path, page_num, dpi=300):
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    pix = page.get_pixmap(dpi=dpi)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    return img, (pix.width, pix.height)

def extract_text_with_pymupdf(pdf_path, page_num, dpi=300):
    """Extract text using PyMuPDF with accurate coordinate scaling"""
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Get page dimensions
    page_rect = page.rect
    page_width_pts = page_rect.width
    page_height_pts = page_rect.height
    
    # Calculate scaling factor from points to pixels at given DPI
    scale_factor = dpi / 72.0  # 72 points per inch
    
    # Get text blocks with detailed information
    text_blocks = page.get_text("dict")
    elements = []
    
    for block in text_blocks["blocks"]:
        if "lines" in block:  # Text block
            for line in block["lines"]:
                line_text = ""
                line_bbox = None
                font_sizes = []
                
                for span in line["spans"]:
                    line_text += span["text"]
                    font_sizes.append(span["size"])
                    
                    # Get span bounding box in points
                    span_bbox = span["bbox"]
                    
                    # Calculate line bounding box
                    if line_bbox is None:
                        line_bbox = list(span_bbox)
                    else:
                        line_bbox[0] = min(line_bbox[0], span_bbox[0])  # min x
                        line_bbox[1] = min(line_bbox[1], span_bbox[1])  # min y
                        line_bbox[2] = max(line_bbox[2], span_bbox[2])  # max x
                        line_bbox[3] = max(line_bbox[3], span_bbox[3])  # max y
                
                if line_text.strip() and line_bbox:
                    # Scale coordinates from points to pixels
                    scaled_bbox = [
                        line_bbox[0] * scale_factor,  # x0
                        line_bbox[1] * scale_factor,  # y0
                        line_bbox[2] * scale_factor,  # x1
                        line_bbox[3] * scale_factor   # y1
                    ]
                    
                    avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12
                    elements.append({
                        'type': 'text',
                        'bbox': tuple(scaled_bbox),
                        'text': line_text.strip(),
                        'font_size': avg_font_size * scale_factor  # Scale font size too
                    })
    
    return elements



def parse_pdf_native(pdf_path, page_num, dpi=300):
    """Simplified PDF parsing focusing on accurate text extraction"""
    elements = []
    
    # Use PyMuPDF for text extraction with proper coordinate scaling
    try:
        pymupdf_elements = extract_text_with_pymupdf(pdf_path, page_num, dpi)
        elements.extend(pymupdf_elements)
        print(f"Extracted {len(pymupdf_elements)} text elements from PDF")
    except Exception as e:
        print(f"Error in PyMuPDF text extraction: {e}")
    
    # Extract images using PyMuPDF
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num]
        scale_factor = dpi / 72.0
        
        image_list = page.get_images(full=True)
        for img_info in image_list:
            xref = img_info[0]
            try:
                # Get image rectangle on page
                img_rects = page.get_image_rects(xref)
                if img_rects:
                    for rect in img_rects:
                        # Scale coordinates from points to pixels
                        scaled_bbox = [
                            rect.x0 * scale_factor,
                            rect.y0 * scale_factor, 
                            rect.x1 * scale_factor,
                            rect.y1 * scale_factor
                        ]
                        elements.append({
                            'type': 'image',
                            'bbox': tuple(scaled_bbox),
                            'xref': xref
                        })
            except Exception as e:
                print(f"Error processing image xref {xref}: {e}")
                
    except Exception as e:
        print(f"Error in image extraction: {e}")
    
    return elements