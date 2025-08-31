import sys
import os
sys.path.append('src')

# Test basic imports
try:
    from parsers.pdf_parser import parse_pdf_native
    from detectors.vision_detectors import load_block_detector
    print("✅ All imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Test PDF parsing
try:
    if os.path.exists('sample.pdf'):
        elements = parse_pdf_native('sample.pdf', 0)
        text_elements = [e for e in elements if e['type'] == 'text']
        print(f"✅ PDF parsing successful: found {len(text_elements)} text elements")
        
        # Show first few text elements
        for i, elem in enumerate(text_elements[:3]):
            print(f"   Text {i+1}: '{elem['text'][:50]}...'")
    else:
        print("⚠️  No sample.pdf found for testing")
except Exception as e:
    print(f"❌ PDF parsing error: {e}")

print("Quick test completed!")