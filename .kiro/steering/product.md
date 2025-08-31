# Product Overview

This is a PDF layout analysis and document understanding system that combines computer vision models with native PDF parsing to extract and classify document elements.

## Core Functionality

- **Layout Detection**: Uses DETR-based models to detect document layout elements (text blocks, tables, images, headers, etc.)
- **Multi-Modal Processing**: Combines vision-based detection with native PDF text extraction
- **Cross-Page Analysis**: Detects headers/footers and maintains document structure across pages
- **Element Fusion**: Merges and refines detection results from multiple sources
- **Caption Linking**: Associates captions with their corresponding figures and tables

## Key Features

- Fine-tuning capabilities for DETR models on DocLayNet dataset
- Configurable confidence thresholds and processing parameters
- Visual output generation for debugging and validation
- JSON export of structured document analysis results
- Reading order preservation for extracted elements

The system is designed for document digitization, content extraction, and automated document processing workflows.