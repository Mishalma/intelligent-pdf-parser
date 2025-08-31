# Project Structure

## Root Directory

- `fine_tune_detr.py`: Standalone script for fine-tuning DETR models on DocLayNet dataset
- `test.py`: Simple model loading test script
- `*.pdf`: Sample PDF files for testing (resume.pdf, sample.pdf, samplenew.pdf)
- `.venv/`: Python virtual environment
- `doclaynet_parquet_info.json`: Dataset metadata

## Source Code Organization (`src/`)

### Main Entry Point
- `src/main.py`: Primary processing pipeline that orchestrates the entire document analysis workflow

### Configuration
- `src/configs/models.yaml`: Model configurations, confidence thresholds, and processing parameters

### Core Modules

#### Parsers (`src/parsers/`)
- `pdf_parser.py`: PDF rendering and native text extraction functionality

#### Detectors (`src/detectors/`)
- `vision_detectors.py`: Computer vision model loading and inference for layout detection

#### Fusion (`src/fusion/`)
- `fusion.py`: Core logic for merging detection results from multiple sources
- `cross_page.py`: Header/footer detection across document pages
- `caption_linker.py`: Logic for associating captions with figures and tables

#### Utilities (`src/utils/`)
- `output.py`: Result serialization (JSON) and visualization generation

#### Output (`src/output/`)
- Directory for generated results, visualizations, and processed outputs

## Architecture Patterns

- **Modular Design**: Clear separation between parsing, detection, fusion, and output
- **Configuration-Driven**: YAML-based configuration for model parameters
- **Pipeline Architecture**: Sequential processing with intermediate result fusion
- **Multi-Modal Processing**: Combines vision and text-based approaches

## File Naming Conventions

- Snake_case for Python files and directories
- Descriptive module names reflecting functionality
- Configuration files use `.yaml` extension
- Output files include page numbers and descriptive names