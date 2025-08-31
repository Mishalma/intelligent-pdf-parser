#!/usr/bin/env python3

import sys
import os
sys.path.append('src')

from main import main

def test_page_1_ultra_aggressive():
    """Test ultra-aggressive text removal on page 1 (the page with the table)"""
    
    print("======================================================================")
    print("TESTING ULTRA-AGGRESSIVE TEXT REMOVAL ON PAGE 1")
    print("======================================================================")
    
    # Test page 1 which has the table
    sys.argv = ['main.py', '--pdf', 'sample.pdf', '--page', '1']
    
    try:
        main()
        print("✅ Ultra-aggressive text removal test completed!")
        print("Check outputs/enhanced_detection_page_1.png to see the results")
        print("Text boxes inside the table should now be removed with 10% overlap threshold")
    except Exception as e:
        print(f"❌ Error during test: {e}")
        raise

if __name__ == "__main__":
    test_page_1_ultra_aggressive()