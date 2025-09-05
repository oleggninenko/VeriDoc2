#!/usr/bin/env python3
"""
Test script for the new Excel structure with fixed first two columns and dynamic output fields.
This script tests the new Excel generation functionality.
"""

import pandas as pd
import tempfile
import os
import json
from verify_statements import excel_prepare_court_writer

def test_excel_writer_with_output_fields():
    """Test Excel writer with output fields configuration"""
    print("Testing Excel writer with output fields...")
    
    # Test output fields configuration
    output_fields = [
        {"id": "par_number", "name": "Paragraph Number", "description": "Paragraph number (fixed)", "enabled": True, "fixed": True},
        {"id": "par_context", "name": "Statement", "description": "Statement content (fixed)", "enabled": True, "fixed": True},
        {"id": "is_accurate", "name": "Is Accurate", "description": "Accuracy verdict", "enabled": True, "fixed": False},
        {"id": "degree_accuracy", "name": "Degree of Accuracy", "description": "Accuracy score", "enabled": True, "fixed": False},
        {"id": "custom_field", "name": "Custom Analysis", "description": "Custom analysis result", "enabled": True, "fixed": False}
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        filename = tmp.name
    
    try:
        # Test Excel writer
        output_path, wb = excel_prepare_court_writer(filename, output_fields)
        
        # Verify the workbook structure
        ws = wb.active
        headers = [cell.value for cell in ws[1]]
        
        expected_headers = ["Paragraph Number", "Statement", "Is Accurate", "Degree of Accuracy", "Custom Analysis"]
        
        print(f"Generated headers: {headers}")
        print(f"Expected headers: {expected_headers}")
        
        if headers == expected_headers:
            print("✅ Excel writer test passed. Headers match expected structure.")
            return True
        else:
            print("❌ Excel writer test failed. Headers don't match expected structure.")
            return False
            
    except Exception as e:
        print(f"❌ Excel writer test failed with error: {e}")
        return False
    finally:
        os.unlink(filename)

def test_excel_writer_without_output_fields():
    """Test Excel writer without output fields (should still have fixed columns)"""
    print("\nTesting Excel writer without output fields...")
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        filename = tmp.name
    
    try:
        # Test Excel writer without output fields
        output_path, wb = excel_prepare_court_writer(filename, None)
        
        # Verify the workbook structure
        ws = wb.active
        headers = [cell.value for cell in ws[1]]
        
        expected_headers = ["Paragraph Number", "Statement"]
        
        print(f"Generated headers: {headers}")
        print(f"Expected headers: {expected_headers}")
        
        if headers == expected_headers:
            print("✅ Excel writer test without output fields passed.")
            return True
        else:
            print("❌ Excel writer test without output fields failed.")
            return False
            
    except Exception as e:
        print(f"❌ Excel writer test without output fields failed with error: {e}")
        return False
    finally:
        os.unlink(filename)

def test_excel_writer_with_disabled_fields():
    """Test Excel writer with some disabled output fields"""
    print("\nTesting Excel writer with disabled fields...")
    
    # Test output fields with some disabled
    output_fields = [
        {"id": "par_number", "name": "Paragraph Number", "description": "Paragraph number (fixed)", "enabled": True, "fixed": True},
        {"id": "par_context", "name": "Statement", "description": "Statement content (fixed)", "enabled": True, "fixed": True},
        {"id": "is_accurate", "name": "Is Accurate", "description": "Accuracy verdict", "enabled": False, "fixed": False},
        {"id": "degree_accuracy", "name": "Degree of Accuracy", "description": "Accuracy score", "enabled": True, "fixed": False},
        {"id": "custom_field", "name": "Custom Analysis", "description": "Custom analysis result", "enabled": False, "fixed": False}
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        filename = tmp.name
    
    try:
        # Test Excel writer
        output_path, wb = excel_prepare_court_writer(filename, output_fields)
        
        # Verify the workbook structure
        ws = wb.active
        headers = [cell.value for cell in ws[1]]
        
        expected_headers = ["Paragraph Number", "Statement", "Degree of Accuracy"]
        
        print(f"Generated headers: {headers}")
        print(f"Expected headers: {expected_headers}")
        
        if headers == expected_headers:
            print("✅ Excel writer test with disabled fields passed.")
            return True
        else:
            print("❌ Excel writer test with disabled fields failed.")
            return False
            
    except Exception as e:
        print(f"❌ Excel writer test with disabled fields failed with error: {e}")
        return False
    finally:
        os.unlink(filename)

def test_output_fields_config():
    """Test the output fields configuration file"""
    print("\nTesting output fields configuration...")
    
    try:
        # Check if the config file exists and has the correct structure
        if os.path.exists("output_fields_config.json"):
            with open("output_fields_config.json", 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"Loaded {len(config)} fields from config")
            
            # Check for fixed columns
            fixed_fields = [f for f in config if f.get("fixed", False)]
            print(f"Fixed fields: {[f['name'] for f in fixed_fields]}")
            
            # Check for dynamic fields
            dynamic_fields = [f for f in config if not f.get("fixed", False)]
            print(f"Dynamic fields: {[f['name'] for f in dynamic_fields]}")
            
            # Verify fixed columns exist and have correct names
            par_number_field = next((f for f in config if f['id'] == 'par_number'), None)
            par_context_field = next((f for f in config if f['id'] == 'par_context'), None)
            
            if (par_number_field and par_number_field['name'] == 'Paragraph Number' and 
                par_context_field and par_context_field['name'] == 'Statement'):
                print("✅ Output fields configuration test passed.")
                return True
            else:
                print("❌ Output fields configuration test failed. Fixed columns not properly configured.")
                return False
        else:
            print("❌ Output fields configuration file not found.")
            return False
            
    except Exception as e:
        print(f"❌ Output fields configuration test failed with error: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Testing New Excel Structure with Fixed First Two Columns\n")
    
    tests = [
        test_excel_writer_with_output_fields,
        test_excel_writer_without_output_fields,
        test_excel_writer_with_disabled_fields,
        test_output_fields_config
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! New Excel structure is working correctly.")
        print("\n✅ Features verified:")
        print("  - Fixed first two columns (Paragraph Number, Statement)")
        print("  - Dynamic columns based on output fields")
        print("  - Proper handling of enabled/disabled fields")
        print("  - Correct output fields configuration")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
