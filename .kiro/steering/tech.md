# Technology Stack

## Core Dependencies

- **Python**: Primary language
- **PyTorch**: Deep learning framework
- **Transformers**: Hugging Face library for pre-trained models
- **PyMuPDF (fitz)**: PDF processing and rendering
- **PIL/Pillow**: Image processing
- **Albumentations**: Image augmentation for training
- **Datasets**: Hugging Face datasets library
- **NumPy**: Numerical computing
- **YAML**: Configuration management

## Key Models & Frameworks

- **DETR (Detection Transformer)**: Primary layout detection model
- **DocLayNet**: Training dataset for document layout
- **Table Transformer**: Specialized table detection
- **LayoutLMv3**: Document understanding model

## Environment Setup

- Uses Python virtual environment (`.venv/` directory)
- Configuration managed via YAML files in `src/configs/`
- Model parameters and thresholds configurable

## Common Commands

```bash
# Process a PDF document
python src/main.py --pdf path/to/document.pdf

# Fine-tune DETR model on DocLayNet
python fine_tune_detr.py

# Test model loading
python test.py
```

## Configuration

- Model settings: `src/configs/models.yaml`
- Confidence thresholds, DPI settings, and processing parameters are configurable
- Output directory: `src/output/` (results and visualizations)