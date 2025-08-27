"""
Text extraction utilities for various file formats.
Consolidated from duplicate implementations across the codebase.
"""

import os
import PyPDF2
from docx import Document
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def extract_text_from_file(file_path: str) -> str:
    """
    Extract text from a file based on its extension.
    
    Args:
        file_path (str): Path to the file
        
    Returns:
        str: Extracted text content
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_word(file_path)
    elif file_extension in ['.xlsx', '.xls']:
        return extract_text_from_excel(file_path)
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text content
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
        raise


def extract_text_from_word(docx_path: str) -> str:
    """
    Extract text from a Word document.
    
    Args:
        docx_path (str): Path to the Word document
        
    Returns:
        str: Extracted text content
    """
    try:
        doc = Document(docx_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from Word document {docx_path}: {str(e)}")
        raise


def extract_text_from_excel(excel_path: str) -> str:
    """
    Extract text from an Excel file.
    
    Args:
        excel_path (str): Path to the Excel file
        
    Returns:
        str: Extracted text content
    """
    try:
        # Read all sheets
        excel_file = pd.ExcelFile(excel_path)
        text = ""
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            text += f"Sheet: {sheet_name}\n"
            text += df.to_string(index=False) + "\n\n"
        
        return text.strip()
    except Exception as e:
        logger.error(f"Error extracting text from Excel file {excel_path}: {str(e)}")
        raise

