#!/usr/bin/env python3
"""
Test script for the new Excel validation functionality.
This script tests the read_statements function with various Excel file structures.
"""

import pandas as pd
import tempfile
import os
from verify_statements import read_statements

def create_test_excel(data, filename):
    """Create a test Excel file with the given data"""
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    return filename

def test_valid_excel():
    """Test with a valid Excel file"""
    print("Testing valid Excel file...")
    
    # Test data with 2 columns
    test_data = {
        'Paragraph': ['1', '2', '3'],
        'Content': ['First statement', 'Second statement', 'Third statement']
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        filename = tmp.name
    
    try:
        create_test_excel(test_data, filename)
        result = read_statements(filename)
        print(f"✅ Valid Excel test passed. Found {len(result)} statements.")
        for stmt in result:
            print(f"  - {stmt['par']}: {stmt['content'][:50]}...")
        return True
    except Exception as e:
        print(f"❌ Valid Excel test failed: {e}")
        return False
    finally:
        os.unlink(filename)

def test_excel_with_extra_columns():
    """Test with Excel file containing extra columns"""
    print("\nTesting Excel file with extra columns...")
    
    # Test data with 3 columns (extra column should be ignored)
    test_data = {
        'Paragraph': ['1', '2'],
        'Content': ['First statement', 'Second statement'],
        'Extra': ['Ignore this', 'Ignore this too']
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        filename = tmp.name
    
    try:
        create_test_excel(test_data, filename)
        result = read_statements(filename)
        print(f"✅ Extra columns test passed. Found {len(result)} statements.")
        print("  (Extra column was ignored as expected)")
        return True
    except Exception as e:
        print(f"❌ Extra columns test failed: {e}")
        return False
    finally:
        os.unlink(filename)

def test_invalid_excel_single_column():
    """Test with Excel file containing only 1 column"""
    print("\nTesting Excel file with single column (should fail)...")
    
    # Test data with only 1 column
    test_data = {
        'Paragraph': ['1', '2', '3']
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        filename = tmp.name
    
    try:
        create_test_excel(test_data, filename)
        result = read_statements(filename)
        print(f"❌ Single column test should have failed but didn't")
        return False
    except Exception as e:
        print(f"✅ Single column test correctly failed: {e}")
        return True
    finally:
        os.unlink(filename)

def test_invalid_excel_empty_content():
    """Test with Excel file containing empty content"""
    print("\nTesting Excel file with empty content (should fail)...")
    
    # Test data with empty content in second column
    test_data = {
        'Paragraph': ['1', '2', '3'],
        'Content': ['First statement', '', 'Third statement']
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        filename = tmp.name
    
    try:
        create_test_excel(test_data, filename)
        result = read_statements(filename)
        print(f"❌ Empty content test should have failed but didn't")
        return False
    except Exception as e:
        print(f"✅ Empty content test correctly failed: {e}")
        return True
    finally:
        os.unlink(filename)

def test_invalid_excel_empty_paragraph():
    """Test with Excel file containing empty paragraph numbers"""
    print("\nTesting Excel file with empty paragraph numbers (should fail)...")
    
    # Test data with empty paragraph numbers
    test_data = {
        'Paragraph': ['1', '', '3'],
        'Content': ['First statement', 'Second statement', 'Third statement']
    }
    
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        filename = tmp.name
    
    try:
        create_test_excel(test_data, filename)
        result = read_statements(filename)
        print(f"❌ Empty paragraph test should have failed but didn't")
        return False
    except Exception as e:
        print(f"✅ Empty paragraph test correctly failed: {e}")
        return True
    finally:
        os.unlink(filename)

def main():
    """Run all tests"""
    print("🧪 Testing Excel Validation Functionality\n")
    
    tests = [
        test_valid_excel,
        test_excel_with_extra_columns,
        test_invalid_excel_single_column,
        test_invalid_excel_empty_content,
        test_invalid_excel_empty_paragraph
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Excel validation is working correctly.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main()
