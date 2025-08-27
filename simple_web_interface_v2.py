#!/usr/bin/env python3
"""
Simple Web Interface for Document Verification System
This provides a basic web interface without requiring Node.js
"""

import os
import json
import asyncio
import threading
import time
import logging
from pathlib import Path
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('veridoc.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Custom exceptions
class VerificationError(Exception):
    """Custom exception for verification-related errors"""
    pass

class FileProcessingError(Exception):
    """Custom exception for file processing errors"""
    pass

class CacheError(Exception):
    """Custom exception for cache-related errors"""
    pass

# Import your existing verification script
from verify_statements import (
    load_api_credentials, build_or_load_index, read_statements,
    process_statements_parallel, excel_prepare_court_writer,
    excel_append_row, sort_excel_by_paragraph_number,
    list_available_caches, delete_cache, load_multiple_caches,
    run as run_verification
)

# Import shared utilities
from utils.text_extraction import (
    extract_text_from_file, extract_text_from_pdf,
    extract_text_from_word, extract_text_from_excel
)
from utils.embeddings import embed_texts_batch, embed_single_text
from utils.status import (
    update_status, update_deep_analysis_status,
    get_processing_status, get_deep_analysis_status,
    reset_processing_status, reset_deep_analysis_status,
    processing_status, deep_analysis_status
)

app = FastAPI(title="VeriDoc AI API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Global variables for uploaded files
knowledge_files = []
statements_file = None

# Server-side category storage
cache_categories = {}
all_categories = []
CATEGORIES_FILE = Path("cache_categories.json")
CATEGORY_LIST_FILE = Path("category_list.json")

# Default categories
DEFAULT_CATEGORIES = [
    "Uncategorized",
    "A - Principal case documents",
    "AA - Trial Documents", 
    "B - Factual witness statements",
    "C - Law expert reports",
    "D - Forensic and valuation reports",
    "Hearing Transcripts",
    "Orders & Judgements",
    "Other"
]

# Load existing categories from file
def load_cache_categories():
    """Load cache categories from file"""
    global cache_categories, all_categories
    try:
        if CATEGORIES_FILE.exists():
            with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
                cache_categories = json.load(f)
            logger.info(f"Loaded {len(cache_categories)} cache categories from file")
        else:
            cache_categories = {}
            logger.info("No existing cache categories file found, starting fresh")
            
        # Load category list
        if CATEGORY_LIST_FILE.exists():
            with open(CATEGORY_LIST_FILE, 'r', encoding='utf-8') as f:
                all_categories = json.load(f)
            logger.info(f"Loaded {len(all_categories)} categories from category list")
        else:
            all_categories = DEFAULT_CATEGORIES.copy()
            save_category_list()
            logger.info("No existing category list found, using default categories")
            
    except Exception as e:
        logger.error(f"Error loading cache categories: {e}")
        cache_categories = {}
        all_categories = DEFAULT_CATEGORIES.copy()

def save_cache_categories():
    """Save cache categories to file"""
    try:
        with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_categories, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(cache_categories)} cache categories to file")
    except Exception as e:
        logger.error(f"Error saving cache categories: {e}")

def save_category_list():
    """Save category list to file"""
    try:
        with open(CATEGORY_LIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_categories, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(all_categories)} categories to category list")
    except Exception as e:
        logger.error(f"Error saving category list: {e}")

# Load categories on startup
load_cache_categories()

# File storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)







def process_statement_batch(client, judge_model: str, embedding_model: str, 
                           statements_batch: list, embeddings: np.ndarray, 
                           chunks: list, top_k: int = 10, max_snippet_chars: int = 4000) -> list:
    """Process a batch of statements in parallel"""
    results = []
    
    # Prepare statements for embedding
    statements_texts = []
    for stmt in statements_batch:
        # Create three-line context
        content = stmt['content']
        three_line_content = content
        statements_texts.append(three_line_content)
    
    # Batch embed all statements
    batch_embeddings = embed_texts_batch(client, embedding_model, statements_texts)
    
    # Process each statement in the batch
    for i, stmt in enumerate(statements_batch):
        try:
            # Get embedding for this statement
            statement_embedding = batch_embeddings[i]
            
            # Find most similar chunks
            similarities = cosine_similarity([statement_embedding], embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            evidence_snippets = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Only include relevant chunks
                    chunk_text = chunks[idx][1] if isinstance(chunks[idx], tuple) else chunks[idx]
                    evidence_snippets.append(chunk_text)
            
            # Judge the statement
            from verify_statements import judge_court_statement
            result = judge_court_statement(client, judge_model, stmt['par'], statements_texts[i], evidence_snippets)
            
            # Prepare result
            result_dict = {
                'par_number': stmt['par'],
                'par_context': statements_texts[i],
                'is_accurate': result['verdict'],
                'degree_of_accuracy': result['degree_of_accuracy'],
                'inaccuracy_type': result['inaccuracy_type'],
                'description': result['description']
            }
            
            results.append(result_dict)
            
        except Exception as e:
            # Handle errors gracefully
            result_dict = {
                'par_number': stmt['par'],
                'par_context': stmt['content'],
                'is_accurate': 'error',
                'degree_of_accuracy': 0,
                'inaccuracy_type': 'error',
                'description': f'Processing error: {str(e)}'
            }
            results.append(result_dict)
    
    return results


def process_statements_parallel(client, judge_model: str, embedding_model: str,
                               statements: list, embeddings: np.ndarray,
                               chunks: list, top_k: int = 10, max_snippet_chars: int = 4000,
                               max_workers: int = 8) -> list:
    """Process statements in parallel using ThreadPoolExecutor"""
    all_results = []
    batch_size = 10  # Optimal batch size for parallel processing
    
    # Split statements into batches
    batches = []
    for i in range(0, len(statements), batch_size):
        batch = statements[i:i + batch_size]
        batches.append(batch)
    
    update_status("processing", 60, f"Processing {len(statements)} statements in {len(batches)} batches", 0, len(statements))
    
    # Process batches in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for batch in batches:
            future = executor.submit(
                process_statement_batch,
                client, judge_model, embedding_model, batch,
                embeddings, chunks, top_k, max_snippet_chars
            )
            futures.append(future)
        
        # Collect results with progress updates
        completed_batches = 0
        for future in as_completed(futures):
            try:
                batch_results = future.result()
                all_results.extend(batch_results)
                completed_batches += 1
                
                # Update progress
                completed_statements = len(all_results)
                progress = 60 + (completed_statements / len(statements)) * 30
                update_status("processing", progress, 
                            f"Completed batch {completed_batches}/{len(batches)}", 
                            completed_statements, len(statements))
                
            except Exception as e:
                update_status("processing", progress, f"Batch error: {str(e)}", 
                            len(all_results), len(statements), log=f"Batch processing error: {str(e)}")
    
    return all_results

async def process_verification_background(selected_caches: list, statements_filename: str, selected_statements: list = None):
    """Background task for processing verification"""
    try:
        update_status("processing", 0, "Initializing...", 0, 0, "Starting verification process")
        
        # Load API credentials
        update_status("processing", 5, "Loading API credentials...", 0, 0)
        api_key, base_url = load_api_credentials("api.txt")
        update_status("processing", 10, "API credentials loaded", 0, 0)
        
        # Load statements
        update_status("processing", 15, "Loading statements file...", 0, 0)
        statements_path = UPLOAD_DIR / statements_filename
        all_statements = read_statements(str(statements_path))
        
        # Filter statements based on selection
        if selected_statements:
            statements = [all_statements[i] for i in selected_statements if i < len(all_statements)]
        else:
            statements = all_statements
            
        update_status("processing", 20, f"Loaded {len(statements)} statements", 0, len(statements))
        
        # Load caches or process knowledge files
        update_status("processing", 25, "Loading knowledge database...", 0, len(statements))
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        if selected_caches:
            # Use existing caches
            embeddings, chunks, cache_names = load_multiple_caches(selected_caches)
            update_status("processing", 50, "Knowledge database loaded from caches", 0, len(statements))
        else:
            # Process first knowledge file (for now, we'll use the first one)
            # In a full implementation, you'd combine multiple files
            main_source_file = UPLOAD_DIR / knowledge_files[0]['name']
            embeddings, chunks, index_id = build_or_load_index(
                corpus_path=str(main_source_file),
                embedding_model="text-embedding-3-large",
                client=client,
                chunk_size_chars=4000,
                overlap_chars=500,
            )
            update_status("processing", 50, "Knowledge database built", 0, len(statements))
        
        # Process statements using parallel processing
        update_status("processing", 60, "Starting parallel processing...", 0, len(statements))
        
        # Use parallel processing for better performance
        results = process_statements_parallel(
            client=client,
            judge_model="gpt-5-mini",
            embedding_model="text-embedding-3-large",
            statements=statements,
            embeddings=embeddings,
            chunks=chunks,
            top_k=10,
            max_snippet_chars=4000,  # Use full chunk size
            max_workers=8  # Adjust based on your system capabilities
        )
        
        # Save results to Excel
        update_deep_analysis_status("processing", 90, "Saving results...", len(statements), len(statements))
        
        df = pd.DataFrame(results)
        output_file = "Deep Verification results.xlsx"
        df.to_excel(output_file, index=False)
        
        update_deep_analysis_status("completed", 100, "Deep analysis completed", len(statements), len(statements),
                                   log=f"Deep analysis completed. Results saved to {output_file}")
        
    except Exception as e:
        update_deep_analysis_status("error", 0, "Deep analysis failed", 0, 0, 
                                   log=f"Deep analysis failed: {str(e)}")
        logger.error(f"Deep analysis error: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main HTML page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeriDoc AI</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
                  <div class="header">
              <h1>VeriDoc AI</h1>
              <p>Automated Fact Verification and Document Consistency Platform</p>
          </div>
        
        <div class="content">
                         <div class="tab-container">
                 <button class="tab active" data-tab="setup">Setup</button>
                 <button class="tab" data-tab="results">Results</button>
                 <button class="tab" data-tab="technical">Technical Info</button>
             </div>
            
            <div id="setup" class="tab-content active">
                <!-- Ask Me Zone -->
                <div class="ask-me-section">
                    <h3>🤔 Ask Me</h3>
                    <p>Ask questions about your knowledge database and get AI-powered answers. You can ask follow-up questions and upload additional documents for context.</p>
                    
                    <div class="ask-me-split-container">
                        <!-- Left Part: Question, Buttons, and Context Files -->
                        <div class="ask-me-left">
                    <div class="ask-me-content">
                        <!-- Row 1: Input Field -->
                        <div class="ask-me-row">
                            <div class="ask-me-input-area">
                                <label for="askMeInput">💬 Question:</label>
                                        <br>
                                <textarea id="askMeInput" placeholder="Type your question here... You can ask follow-up questions based on previous answers."></textarea>
                            </div>
                        </div>
                        
                        <!-- Row 2: Buttons -->
                        <div class="ask-me-row">
                            <div class="ask-me-buttons">
                                <button class="btn primary" id="askMeBtn" disabled>💬 Ask Question</button>
                                <button class="btn secondary small" id="uploadContextBtn">📎 Add Context</button>
                                <button class="btn secondary small" id="newConversationBtn">🆕 New Conversation</button>
                            </div>
                        </div>
                        
                        <!-- Hidden file input -->
                        <input type="file" id="askMeFileInput" multiple accept=".pdf,.docx,.doc,.xlsx,.xls,.txt" class="file-input-hidden" aria-label="Upload context files for conversation" title="Upload context files for conversation">
                        
                        <!-- Row 3: Context Files -->
                        <div class="ask-me-row">
                            <div id="askMeFileList" class="ask-me-file-list">
                                <div class="empty-state">
                                    <div class="empty-state-text">No context files uploaded yet</div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <!-- Right Part: Conversation History -->
                        <div class="ask-me-right">
                            <div class="conversation-history" id="conversationHistory">
                                <div class="conversation-header">
                                    <h4>💬 Conversation History</h4>
                                    <button class="conversation-copy-btn" onclick="copyWholeConversation()" title="Copy entire conversation">📋</button>
                                </div>
                                <div id="conversationMessages" class="conversation-messages">
                                    <div class="empty-state">
                                        <div class="empty-state-icon">💭</div>
                                        <div class="empty-state-text">Start a conversation by asking a question...</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="zones-container">
                    <!-- Knowledge Database Zone -->
                    <div class="zone-left">
                        <h3>📚 Knowledge Database</h3>
                        <p>Manage your source documents that will be used as the knowledge base for verification.</p>
                        
                        <div class="upload-area" id="knowledgeUploadArea">
                            <p><strong>📁 Upload Knowledge Files</strong></p>
                            <p>Click to add documents to your knowledge database</p>
                            <p><small>Supports: PDF, Word (DOCX, DOC), Excel (XLSX, XLS), Text (TXT)</small></p>
                            <input type="file" id="knowledgeFileInput" multiple accept=".pdf,.docx,.doc,.xlsx,.xls,.txt" class="file-input-hidden" aria-label="Upload knowledge database files" title="Upload knowledge database files">
                        </div>
                        
                                                 <div id="knowledgeFileList" class="knowledge-file-list"></div>
                         
                         <div class="button-group button-group-margin">
                             <button class="btn" id="indexFilesBtn" disabled>🔧 Index & Cache Files</button>
                         </div>
                         
                                                  <div>
                              <h4>Existing Cached Knowledge</h4>
                              <button class="btn" id="loadCachesBtn">🔄 Load Available Caches</button>
                              <button class="btn" id="syncCategoriesBtn">🔄 Sync Categories</button>
                              <button class="btn" id="manageCategoriesBtn">⚙️ Manage Categories</button>
                              <button class="btn" id="bulkAssignBtn">📦 Bulk Assign</button>
                              <button class="btn" id="autoCategorizeBtn">🏷️ Auto-Categorize</button>
                              <button class="btn" id="clearCategoriesBtn">🗂️ Clear Categories</button>
                              <button class="btn danger" id="deleteCachesBtn">🗑️ Delete Selected</button>
                              <div class="caches-container" id="cachesTable"></div>
                          </div>
                          
                          <!-- Category Management Modal -->
                          <div id="categoryModal" class="modal hidden">
                              <div class="modal-content">
                                  <div class="modal-header">
                                      <h3>Category Management</h3>
                                      <span class="close" onclick="closeCategoryModal()">&times;</span>
                                  </div>
                                  <div class="modal-body">
                                      <div class="category-actions">
                                          <div class="add-category-section">
                                              <h4>Add New Category</h4>
                                              <div class="input-group">
                                                  <input type="text" id="newCategoryName" placeholder="Enter category name" class="form-input">
                                                  <button class="btn btn-primary" onclick="addCategory()">Add Category</button>
                                              </div>
                                          </div>
                                          
                                          <div class="category-list-section">
                                              <h4>Manage Categories</h4>
                                              <div class="search-box">
                                                  <input type="text" id="categorySearch" placeholder="Search categories..." class="form-input" onkeyup="filterCategories()">
                                              </div>
                                              <div id="categoryList" class="category-list">
                                                  <!-- Categories will be loaded here -->
                                              </div>
                                          </div>
                                      </div>
                                  </div>
                              </div>
                          </div>
                          
                          <!-- Bulk Assignment Modal -->
                          <div id="bulkModal" class="modal hidden">
                              <div class="modal-content">
                                  <div class="modal-header">
                                      <h3>Bulk Category Assignment</h3>
                                      <span class="close" onclick="closeBulkModal()">&times;</span>
                                  </div>
                                  <div class="modal-body">
                                      <div class="bulk-assignment-section">
                                          <h4>Assign Category to Selected Documents</h4>
                                          <div class="input-group">
                                              <select id="bulkCategorySelect" class="form-select">
                                                  <option value="">Select a category...</option>
                                              </select>
                                              <button class="btn btn-primary" onclick="performBulkAssignment()">Assign Category</button>
                                          </div>
                                          <div id="bulkSelectionInfo" class="selection-info">
                                              <!-- Selection info will be shown here -->
                                          </div>
                                      </div>
                                  </div>
                              </div>
                          </div>
                    </div>
                    
                    <!-- Statements for Verification Zone -->
                    <div class="zone-right">
                        <div class="statements-section">
                            <h3>📋 Statements for Verification</h3>
                            <p>Upload the Excel file containing statements/paragraphs that need to be verified against your knowledge database.</p>
                            
                            <div class="upload-area statements" id="statementsUploadArea">
                                <p><strong>📊 Upload Statements File</strong></p>
                                <p>Upload Excel file with statements to verify</p>
                                <p><small>Supports: Excel files (.xlsx, .xls) with 'Par' and 'Content' columns</small></p>
                                <input type="file" id="statementsFileInput" accept=".xlsx,.xls" class="file-input-hidden" aria-label="Upload statements file for verification" title="Upload statements file for verification">
                            </div>
                            
                            <!-- Button Container -->
                            <div class="flex-center">
                                <button id="startBtn" class="btn success large" disabled>Verification Process</button>
                                <button id="deepAnalysisBtn" class="btn warning large" disabled>Deep Analysis</button>
                            </div>
                            
                            <!-- Processing Note (hidden by default) -->
                            <div id="processingNote" class="processing-note processing-note-hidden">
                                Verification process may take several minutes, please keep calm and wait for the verification results
                            </div>
                            
                            <!-- Completion Note (hidden by default) -->
                            <div id="completionNote" class="completion-note completion-note-hidden">
                                ✅ Go to <a href="#" onclick="showTab('results'); return false;" class="results-link">Results tab</a> to check the progress
                            </div>
                            
                            <div id="statementsFileInfo" class="file-list"></div>
                            
                            <!-- Select All Checkbox for Statements -->
                            <div id="statementsSelectAll" class="statements-select-all-container">
                                <label class="flex-label">
                                    <input type="checkbox" id="selectAllStatements" class="checkbox-margin" aria-label="Select all statements for verification" title="Select all statements for verification">
                                    <strong>Select All Statements</strong>
                                </label>
                            </div>
                            
                            <div id="statementsList" class="statements-list statements-list-hidden"></div>
                        </div>
                    </div>
                </div>
            </div>
            
                         <div id="results" class="tab-content">
                 <h2>Results</h2>
                 <button class="btn" id="loadResultsBtn">📊 Load Results</button>
                 <button class="btn success" id="downloadResultsBtn">💾 Download Results</button>
                 <button class="btn warning" id="loadDeepResultsBtn">🔍 Load Deep Results</button>
                 <button class="btn warning" id="downloadDeepResultsBtn">💾 Download Deep Results</button>
                 <div id="resultsTable"></div>
             </div>
             
             <div id="technical" class="tab-content">
                 <h2>🔧 Technical Configuration</h2>
                 <p>Current technical parameters used for document verification processing.</p>
                 
                 <div class="tech-info-grid">
                     <div class="tech-card">
                         <h3>🤖 AI Models</h3>
                         <div class="tech-item">
                             <span class="tech-label">Judge Model:</span>
                             <span class="tech-value">gpt-5-mini</span>
                         </div>
                         <div class="tech-item">
                             <span class="tech-label">Deep Analysis Model:</span>
                             <span class="tech-value">gpt-5-mini</span>
                         </div>
                         <div class="tech-item">
                             <span class="tech-label">Embedding Model:</span>
                             <span class="tech-value">text-embedding-3-large</span>
                         </div>
                     </div>
                     
                     <div class="tech-card">
                         <h3>📄 Text Processing</h3>
                         <div class="tech-item">
                             <span class="tech-label">Chunk Size:</span>
                             <span class="tech-value">4,000 characters</span>
                         </div>
                         <div class="tech-item">
                             <span class="tech-label">Overlap:</span>
                             <span class="tech-value">500 characters</span>
                         </div>
                                                   <div class="tech-item">
                              <span class="tech-label">Max Snippet Length:</span>
                              <span class="tech-value">4,000 characters (full chunk)</span>
                          </div>
                     </div>
                     
                     <div class="tech-card">
                         <h3>🔍 Retrieval Settings</h3>
                         <div class="tech-item">
                             <span class="tech-label">Top-K Results:</span>
                             <span class="tech-value">10 most relevant chunks</span>
                         </div>
                         <div class="tech-item">
                             <span class="tech-label">Similarity Threshold:</span>
                             <span class="tech-value">0.1 (minimum relevance)</span>
                         </div>
                     </div>
                     
                     <div class="tech-card">
                         <h3>⚡ Performance</h3>
                         <div class="tech-item">
                             <span class="tech-label">Parallel Workers:</span>
                             <span class="tech-value">8 concurrent threads</span>
                         </div>
                         <div class="tech-item">
                             <span class="tech-label">Batch Size:</span>
                             <span class="tech-value">10 statements per batch</span>
                         </div>
                         <div class="tech-item">
                             <span class="tech-label">Embedding Batch:</span>
                             <span class="tech-value">256 texts per API call</span>
                         </div>
                     </div>
                     
                     <div class="tech-card">
                         <h3>🧠 AI Processing</h3>
                         <div class="tech-item">
                             <span class="tech-label">Max Tokens:</span>
                             <span class="tech-value">4,000 completion tokens</span>
                         </div>
                         <div class="tech-item">
                             <span class="tech-label">Reasoning Effort:</span>
                             <span class="tech-value">Medium (balanced accuracy/speed)</span>
                         </div>
                         <div class="tech-item">
                             <span class="tech-label">Verbosity:</span>
                             <span class="tech-value">Medium (detailed explanations)</span>
                         </div>
                     </div>
                 </div>
                 
                 <div class="tech-description">
                     <h3>📋 How It Works</h3>
                     <ol>
                         <li><strong>Document Chunking:</strong> Source documents are split into 4,000-character chunks with 500-character overlaps to maintain context.</li>
                         <li><strong>Vector Embedding:</strong> Each chunk is converted to a high-dimensional vector using the text-embedding-3-large model.</li>
                         <li><strong>Statement Processing:</strong> Statements are processed in batches of 10 using 8 parallel workers for efficiency.</li>
                         <li><strong>Similarity Search:</strong> For each statement, the system finds the 10 most similar chunks using cosine similarity.</li>
                         <li><strong>AI Verification:</strong> The gpt-5-mini model analyzes each statement against the relevant evidence chunks with medium reasoning effort and verbosity.</li>
                         <li><strong>Deep Analysis:</strong> The gpt-5-mini model provides comprehensive legal analysis with high reasoning effort and detailed appeal grounds assessment using 8 parallel workers.</li>
                         <li><strong>Response Generation:</strong> AI responses are limited to 4,000 tokens for concise, focused analysis.</li>
                         <li><strong>Accuracy Assessment:</strong> Results include accuracy verdict, degree of accuracy (1-10), and detailed explanations.</li>
                     </ol>
                 </div>
             </div>
        </div>
    </div>
    
    <script>
                 let knowledgeFiles = [];
         let statementsFile = null;
         let selectedCaches = [];
         let selectedStatements = [];
         let statementsData = [];
         let statusPolling = null;
        let deepAnalysisStatusPolling = null;
        
                 // Initialize when DOM is loaded
         document.addEventListener('DOMContentLoaded', function() {
             // Load saved cache names and categories from localStorage
             try {
                 const savedNames = localStorage.getItem('veridoc_cache_names');
                 if (savedNames) {
                     window.cacheNames = JSON.parse(savedNames);
                 }
                 
                 const savedCategories = localStorage.getItem('veridoc_cache_categories');
                 if (savedCategories) {
                     window.cacheCategories = JSON.parse(savedCategories);
                 }
             } catch (error) {
                 console.error('Error loading saved cache data:', error);
             }
             
             // Load custom categories
             loadCustomCategories();
             
             initializeEventListeners();
             loadCaches(); // Auto-load caches on startup
         });
        
        function autoCategorizeCache(cacheName) {
            const name = cacheName.toLowerCase();
            
            // Auto-categorization rules based on common patterns
            if (name.includes('judgment') || name.includes('judgement') || name.includes('approved judgment')) {
                return 'Orders & Judgements';
            }
            if (name.includes('transcript') || name.includes('hearing')) {
                return 'Hearing Transcripts';
            }
            if (name.includes('expert') && (name.includes('report') || name.includes('statement'))) {
                if (name.includes('law') || name.includes('legal')) {
                    return 'C - Law expert reports';
                }
                if (name.includes('forensic') || name.includes('valuation') || name.includes('financial')) {
                    return 'D - Forensic and valuation reports';
                }
                return 'C - Law expert reports';
            }
            if (name.includes('witness') && name.includes('statement')) {
                return 'B - Factual witness statements';
            }
            if (name.includes('trial') || name.includes('submission') || name.includes('opening') || name.includes('closing')) {
                return 'AA - Trial Documents';
            }
            if (name.includes('claim') || name.includes('defence') || name.includes('particulars') || name.includes('reply')) {
                return 'A - Principal case documents';
            }
            if (name.includes('case') && (name.includes('summary') || name.includes('chronology'))) {
                return 'A - Principal case documents';
            }
            
            return 'Uncategorized';
        }
        
        function initializeEventListeners() {
            // Tab switching
            const tabButtons = document.querySelectorAll('.tab-container .tab');
            tabButtons.forEach(btn => {
                btn.addEventListener('click', function() {
                    const tabName = this.getAttribute('data-tab');
                    showTab(tabName);
                });
            });
            
            // Knowledge file upload
            const knowledgeUploadArea = document.getElementById('knowledgeUploadArea');
            const knowledgeFileInput = document.getElementById('knowledgeFileInput');
            if (knowledgeUploadArea && knowledgeFileInput) {
                knowledgeUploadArea.addEventListener('click', function() {
                    knowledgeFileInput.click();
                });
                knowledgeFileInput.addEventListener('change', handleKnowledgeFileSelect);
            }
            
            // Statements file upload
            const statementsUploadArea = document.getElementById('statementsUploadArea');
            const statementsFileInput = document.getElementById('statementsFileInput');
            if (statementsUploadArea && statementsFileInput) {
                statementsUploadArea.addEventListener('click', function() {
                    statementsFileInput.click();
                });
                statementsFileInput.addEventListener('change', handleStatementsFileSelect);
            }
            
            // Buttons
            const startBtn = document.getElementById('startBtn');
            if (startBtn) {
                startBtn.addEventListener('click', startProcessing);
            }
            
            const deepAnalysisBtn = document.getElementById('deepAnalysisBtn');
            if (deepAnalysisBtn) {
                deepAnalysisBtn.addEventListener('click', startDeepAnalysis);
            }
            
            const loadCachesBtn = document.getElementById('loadCachesBtn');
            if (loadCachesBtn) {
                loadCachesBtn.addEventListener('click', loadCaches);
            }
            
            const syncCategoriesBtn = document.getElementById('syncCategoriesBtn');
            if (syncCategoriesBtn) {
                syncCategoriesBtn.addEventListener('click', syncCategoriesToServer);
            }
            
            const manageCategoriesBtn = document.getElementById('manageCategoriesBtn');
            if (manageCategoriesBtn) {
                manageCategoriesBtn.addEventListener('click', openCategoryModal);
            }
            
            const bulkAssignBtn = document.getElementById('bulkAssignBtn');
            if (bulkAssignBtn) {
                bulkAssignBtn.addEventListener('click', openBulkModal);
            }
            
            const deleteCachesBtn = document.getElementById('deleteCachesBtn');
            if (deleteCachesBtn) {
                deleteCachesBtn.addEventListener('click', deleteSelectedCaches);
            }
            
            const autoCategorizeBtn = document.getElementById('autoCategorizeBtn');
            if (autoCategorizeBtn) {
                autoCategorizeBtn.addEventListener('click', autoCategorizeAllCaches);
            }
            
            const clearCategoriesBtn = document.getElementById('clearCategoriesBtn');
            if (clearCategoriesBtn) {
                clearCategoriesBtn.addEventListener('click', clearAllCategories);
            }
            
            const indexFilesBtn = document.getElementById('indexFilesBtn');
            if (indexFilesBtn) {
                indexFilesBtn.addEventListener('click', indexKnowledgeFiles);
            }
            
            const askMeBtn = document.getElementById('askMeBtn');
            if (askMeBtn) {
                askMeBtn.addEventListener('click', askMeQuestion);
            }
            
            const askMeInput = document.getElementById('askMeInput');
            if (askMeInput) {
                askMeInput.addEventListener('input', updateAskMeButtonState);
            }
            
            // Enhanced Ask Me buttons
            
            const newConversationBtn = document.getElementById('newConversationBtn');
            if (newConversationBtn) {
                newConversationBtn.addEventListener('click', newConversation);
            }
            
            const uploadContextBtn = document.getElementById('uploadContextBtn');
            if (uploadContextBtn) {
                uploadContextBtn.addEventListener('click', function() {
                    const askMeFileInput = document.getElementById('askMeFileInput');
                    if (askMeFileInput) {
                        askMeFileInput.click();
                    }
                });
            }
            
            // Initialize file upload for Ask Me
            initializeAskMeFileUpload();
            
            const loadResultsBtn = document.getElementById('loadResultsBtn');
            if (loadResultsBtn) {
                loadResultsBtn.addEventListener('click', loadResults);
            }
            
                         const downloadResultsBtn = document.getElementById('downloadResultsBtn');
             if (downloadResultsBtn) {
                 downloadResultsBtn.addEventListener('click', downloadResults);
             }
             
             const loadDeepResultsBtn = document.getElementById('loadDeepResultsBtn');
             if (loadDeepResultsBtn) {
                 loadDeepResultsBtn.addEventListener('click', loadDeepResults);
             }
             
             const downloadDeepResultsBtn = document.getElementById('downloadDeepResultsBtn');
             if (downloadDeepResultsBtn) {
                 downloadDeepResultsBtn.addEventListener('click', downloadDeepResults);
             }
             
             // Select All Statements checkbox
             const selectAllStatements = document.getElementById('selectAllStatements');
             if (selectAllStatements) {
                 selectAllStatements.addEventListener('change', function() {
                     const checkboxes = document.querySelectorAll('.statement-checkbox');
                     checkboxes.forEach(cb => {
                         cb.checked = this.checked;
                         const index = parseInt(cb.getAttribute('data-index'));
                         if (this.checked) {
                             if (!selectedStatements.includes(index)) {
                                 selectedStatements.push(index);
                             }
                         } else {
                             selectedStatements = selectedStatements.filter(i => i !== index);
                         }
                     });
                     updateStartButtonState();
                 });
             }
        }
        
        function showTab(tabName) {
            // Hide all tab contents
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
            });
            
            // Remove active class from all tabs
            document.querySelectorAll('.tab').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // Show selected tab content
            const tabContent = document.getElementById(tabName);
            if (tabContent) {
                tabContent.classList.add('active');
            }
            
            // Add active class to clicked tab
            const activeTab = document.querySelector('[data-tab="' + tabName + '"]');
            if (activeTab) {
                activeTab.classList.add('active');
            }
            
            // Start/stop status polling
            if (tabName === 'processing') {
                startStatusPolling();
                startDeepAnalysisStatusPolling();
            } else {
                stopStatusPolling();
                stopDeepAnalysisStatusPolling();
            }
        }
        
        function handleKnowledgeFileSelect(event) {
            const files = event.target.files;
            for (let file of files) {
                knowledgeFiles.push({
                    name: file.name,
                    size: file.size,
                    file: file
                });
            }
            updateKnowledgeFileList();
            uploadKnowledgeFiles(files);
        }
        
        function handleStatementsFileSelect(event) {
            const file = event.target.files[0];
            if (file) {
                statementsFile = {
                    name: file.name,
                    size: file.size,
                    file: file
                };
                updateStatementsFileInfo();
                uploadStatementsFile(file);
            }
        }
        
        function updateKnowledgeFileList() {
            const fileList = document.getElementById('knowledgeFileList');
            fileList.innerHTML = '';
            
            if (knowledgeFiles.length === 0) {
                fileList.innerHTML = '<p><em>No knowledge files uploaded yet</em></p>';
                return;
            }
            
            knowledgeFiles.forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                const fileInfo = document.createElement('span');
                fileInfo.textContent = file.name + ' (' + fileSize + ' MB)';
                
                const removeButton = document.createElement('button');
                removeButton.textContent = 'Remove';
                removeButton.className = 'btn danger';
                removeButton.style.padding = '5px 10px';
                removeButton.addEventListener('click', function() {
                    removeKnowledgeFile(index);
                });
                
                fileItem.appendChild(fileInfo);
                fileItem.appendChild(removeButton);
                fileList.appendChild(fileItem);
            });
            
            updateStartButtonState();
        }
        
        function updateStatementsFileInfo() {
            const fileInfo = document.getElementById('statementsFileInfo');
            fileInfo.innerHTML = '';
            
            if (!statementsFile) {
                fileInfo.innerHTML = '<p><em>No statements file uploaded yet</em></p>';
                return;
            }
            
            const fileItem = document.createElement('div');
            fileItem.className = 'file-item';
            
            const fileSize = (statementsFile.size / 1024 / 1024).toFixed(2);
            const fileInfoSpan = document.createElement('span');
            fileInfoSpan.textContent = statementsFile.name + ' (' + fileSize + ' MB)';
            
            const removeButton = document.createElement('button');
            removeButton.textContent = 'Remove';
            removeButton.className = 'btn danger';
            removeButton.style.padding = '5px 10px';
            removeButton.addEventListener('click', function() {
                statementsFile = null;
                updateStatementsFileInfo();
                updateStartButtonState();
            });
            
            fileItem.appendChild(fileInfoSpan);
            fileItem.appendChild(removeButton);
            fileInfo.appendChild(fileItem);
            
            updateStartButtonState();
        }
        
        function removeKnowledgeFile(index) {
            if (index >= 0 && index < knowledgeFiles.length) {
                knowledgeFiles.splice(index, 1);
                updateKnowledgeFileList();
            }
        }
        
                                               function updateStartButtonState() {
                const startBtn = document.getElementById('startBtn');
                const deepAnalysisBtn = document.getElementById('deepAnalysisBtn');
                const indexFilesBtn = document.getElementById('indexFilesBtn');
                
                if (startBtn) {
                    const hasKnowledge = knowledgeFiles.length > 0 || selectedCaches.length > 0;
                    const hasStatements = statementsFile !== null;
                    startBtn.disabled = !(hasKnowledge && hasStatements);
                }
                
                if (deepAnalysisBtn) {
                    const hasKnowledge = knowledgeFiles.length > 0 || selectedCaches.length > 0;
                    const hasStatements = statementsFile !== null;
                    deepAnalysisBtn.disabled = !(hasKnowledge && hasStatements);
                }
                
                if (indexFilesBtn) {
                    indexFilesBtn.disabled = knowledgeFiles.length === 0;
                }
                
                // Update Ask Me button state
                updateAskMeButtonState();
            }
            
            function updateAskMeButtonState() {
                const askMeInput = document.getElementById('askMeInput');
                const askMeBtn = document.getElementById('askMeBtn');
                
                if (askMeInput && askMeBtn) {
                    const hasQuestion = askMeInput.value.trim().length > 0;
                    const hasKnowledge = selectedCaches.length > 0 || currentConversationId;
                    askMeBtn.disabled = !(hasQuestion && hasKnowledge);
                }
            }
            
            // Global variables for conversation
            let conversationHistory = [];
            let additionalContext = "";
            let askMeFiles = [];
            let currentConversationId = null;
            
            async function askMeQuestion() {
                const askMeInput = document.getElementById('askMeInput');
                const askMeBtn = document.getElementById('askMeBtn');
                const conversationMessages = document.getElementById('conversationMessages');
                
                console.log('Elements found:', { askMeInput: !!askMeInput, askMeBtn: !!askMeBtn, conversationMessages: !!conversationMessages });
                
                if (!askMeInput || !askMeBtn || !conversationMessages) {
                    console.error('Required elements not found');
                    return;
                }
                
                const question = askMeInput.value.trim();
                if (!question) return;
                
                // Add user message to conversation
                const timestamp = new Date().toLocaleTimeString();
                addMessageToConversation('user', question, timestamp);
                
                // Show loading animation
                const loadingElement = showLoadingMessage();
                
                // Disable button and show loading
                askMeBtn.disabled = true;
                askMeBtn.textContent = 'Processing...';
                
                try {
                    const response = await fetch('/api/ask-me', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8',
                        },
                        body: JSON.stringify({
                            question: question,
                            selected_caches: selectedCaches,
                            conversation_history: conversationHistory,
                            additional_context: additionalContext,
                            conversation_id: currentConversationId
                        })
                    });
                    
                    console.log('Response status:', response.status);
                    
                    if (response.ok) {
                        const data = await response.json();
                        console.log('Response data:', data);
                        
                        // Remove loading animation
                        removeLoadingMessage(loadingElement);
                        
                        // Add assistant response to conversation
                        const answer = data.answer || 'No answer provided';
                        addMessageToConversation('assistant', answer, timestamp, data);
                        
                        // Update conversation history for next request
                        conversationHistory.push({"role": "user", "content": question});
                        conversationHistory.push({"role": "assistant", "content": answer});
                        
                        // Clear input
                        askMeInput.value = '';
                        
                    } else {
                        const errorData = await response.json();
                        console.log('Error data:', errorData);
                        
                        // Remove loading animation
                        removeLoadingMessage(loadingElement);
                        
                        addMessageToConversation('assistant', `Error: ${errorData.detail}`, timestamp);
                    }
                } catch (error) {
                    console.error('Error asking question:', error);
                    
                    // Remove loading animation
                    removeLoadingMessage(loadingElement);
                    
                    addMessageToConversation('assistant', 'Error: Failed to get answer. Please try again.', timestamp);
                } finally {
                    // Re-enable button
                    askMeBtn.disabled = false;
                    askMeBtn.textContent = '💬 Ask Question';
                }
            }
            
            function addMessageToConversation(role, content, timestamp, data = null) {
                const conversationMessages = document.getElementById('conversationMessages');
                
                // Format content with paragraphs for assistant messages
                let formattedContent = content;
                if (role === 'assistant') {
                    // Split by double newlines to create paragraphs
                    formattedContent = content.split('\\n\\n').map(paragraph => 
                        paragraph.trim() ? `<p>${paragraph.trim()}</p>` : ''
                    ).join('');
                }
                
                let messageHtml = `
                    <div class="message ${role}">
                        <div class="message-time">${timestamp}</div>`;
                
                // Add copy button for assistant messages
                if (role === 'assistant') {
                    // Use data attribute to avoid JavaScript string escaping issues
                    const messageId = 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
                    messageHtml += `<button class="message-copy-btn" data-content="${encodeURIComponent(content)}" onclick="copyMessageContentFromData(this)">📋</button>`;
                }
                
                messageHtml += `<div class="message-content">${formattedContent}</div>`;
                
                if (data && data.confidence) {
                    const confidenceClass = `confidence-${data.confidence}`;
                    messageHtml += `<div class="message-confidence"><strong>Confidence:</strong> <span class="${confidenceClass}">${data.confidence}</span></div>`;
                }
                
                if (data && data.key_points && data.key_points.length > 0) {
                    messageHtml += `<div class="message-key-points"><strong>Key Points:</strong><ul>`;
                    data.key_points.forEach(point => {
                        messageHtml += `<li>${point}</li>`;
                    });
                    messageHtml += `</ul></div>`;
                }
                
                messageHtml += `</div>`;
                
                conversationMessages.innerHTML += messageHtml;
                conversationMessages.scrollTop = conversationMessages.scrollHeight;
            }
            
            function showLoadingMessage() {
                const conversationMessages = document.getElementById('conversationMessages');
                const loadingHtml = `
                    <div class="message assistant">
                        <div class="message-time">${new Date().toLocaleTimeString()}</div>
                        <div class="message-content">
                            <div class="loading-dots">
                                <div></div>
                                <div></div>
                                <div></div>
                                <div></div>
                            </div>
                        </div>
                    </div>
                `;
                conversationMessages.innerHTML += loadingHtml;
                conversationMessages.scrollTop = conversationMessages.scrollHeight;
                return conversationMessages.lastElementChild;
            }
            
            function removeLoadingMessage(loadingElement) {
                if (loadingElement && loadingElement.parentNode) {
                    loadingElement.remove();
                }
            }
            
            function copyMessageContentFromData(button) {
                const content = decodeURIComponent(button.getAttribute('data-content'));
                navigator.clipboard.writeText(content).then(() => {
                    // Visual feedback
                    button.classList.add('copied');
                    button.textContent = '✓';
                    
                    setTimeout(() => {
                        button.classList.remove('copied');
                        button.textContent = '📋';
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy: ', err);
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = content;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    // Visual feedback
                    button.classList.add('copied');
                    button.textContent = '✓';
                    
                    setTimeout(() => {
                        button.classList.remove('copied');
                        button.textContent = '📋';
                    }, 2000);
                });
            }
            
            function copyWholeConversation() {
                const conversationMessages = document.getElementById('conversationMessages');
                const messages = conversationMessages.querySelectorAll('.message');
                
                if (messages.length === 0) {
                    // No messages to copy
                    return;
                }
                
                let conversationText = '';
                
                messages.forEach(message => {
                    const role = message.classList.contains('user') ? 'User' : 'Assistant';
                    const time = message.querySelector('.message-time')?.textContent || '';
                    const content = message.querySelector('.message-content')?.textContent || '';
                    
                    conversationText += `[${time}] ${role}:\n${content}\n\n`;
                });
                
                // Remove trailing newlines
                conversationText = conversationText.trim();
                
                const copyBtn = document.querySelector('.conversation-copy-btn');
                
                navigator.clipboard.writeText(conversationText).then(() => {
                    // Visual feedback
                    copyBtn.classList.add('copied');
                    copyBtn.textContent = '✓';
                    
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.textContent = '📋';
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy conversation: ', err);
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = conversationText;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    // Visual feedback
                    copyBtn.classList.add('copied');
                    copyBtn.textContent = '✓';
                    
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.textContent = '📋';
                    }, 2000);
                });
            }
            

            
            async function newConversation() {
                // Delete context cache if exists
                if (currentConversationId) {
                    try {
                        await fetch('/api/ask-me-delete-context', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json; charset=utf-8',
                            },
                            body: JSON.stringify({
                                conversation_id: currentConversationId
                            })
                        });
                        console.log('Deleted context cache for conversation:', currentConversationId);
                    } catch (error) {
                        console.error('Error deleting context cache:', error);
                    }
                }
                
                // Reset conversation state
                conversationHistory = [];
                additionalContext = "";
                askMeFiles = [];
                currentConversationId = null;
                
                const conversationMessages = document.getElementById('conversationMessages');
                conversationMessages.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">💭</div>
                        <div class="empty-state-text">Start a conversation by asking a question...</div>
                    </div>
                `;
                
                // Clear context files display
                displayAskMeFiles();
                
                const askMeInput = document.getElementById('askMeInput');
                askMeInput.value = '';
                askMeInput.focus();
            }
            
            // Document upload functionality for Ask Me
            function initializeAskMeFileUpload() {
                const uploadContextBtn = document.getElementById('uploadContextBtn');
                const askMeFileInput = document.getElementById('askMeFileInput');
                const askMeFileList = document.getElementById('askMeFileList');
                
                if (uploadContextBtn && askMeFileInput) {
                    // Remove any existing event listeners to prevent duplicates
                    uploadContextBtn.replaceWith(uploadContextBtn.cloneNode(true));
                    const newUploadContextBtn = document.getElementById('uploadContextBtn');
                    newUploadContextBtn.addEventListener('click', () => askMeFileInput.click());
                    
                    askMeFileInput.addEventListener('change', async (event) => {
                        const files = Array.from(event.target.files);
                        if (files.length === 0) return;
                        
                        console.log('Files selected:', files.map(f => f.name));
                        
                        const formData = new FormData();
                        files.forEach(file => {
                            formData.append('files', file);
                            console.log('Added file to form data:', file.name);
                        });
                        
                        // Add conversation ID if available
                        if (currentConversationId) {
                            formData.append('conversation_id', currentConversationId);
                        }
                        
                        try {
                            console.log('Sending upload request...');
                            const response = await fetch('/api/ask-me-upload', {
                                method: 'POST',
                                body: formData
                            });
                            
                            console.log('Upload response status:', response.status);
                            
                            if (response.ok) {
                                const data = await response.json();
                                console.log('Upload response data:', data);
                                askMeFiles = data.uploaded_files;
                                
                                // Store conversation ID if provided
                                if (data.conversation_id) {
                                    currentConversationId = data.conversation_id;
                                    console.log('Conversation ID set to:', currentConversationId);
                                }
                                
                                displayAskMeFiles();
                                updateAdditionalContext();
                                console.log('Files uploaded successfully:', askMeFiles.length);
                            } else {
                                const errorData = await response.json();
                                console.error('Upload failed:', errorData);
                                alert('Upload failed: ' + (errorData.detail || 'Unknown error'));
                            }
                        } catch (error) {
                            console.error('Upload error:', error);
                            alert('Upload error: ' + error.message);
                        }
                    });
                }
            }
            
            function displayAskMeFiles() {
                const askMeFileList = document.getElementById('askMeFileList');
                if (!askMeFileList) return;
                
                // Clear existing content
                askMeFileList.innerHTML = '';
                
                if (askMeFiles.length === 0) {
                    const emptyState = document.createElement('div');
                    emptyState.className = 'empty-state';
                    emptyState.innerHTML = `
                        <div class="empty-state-text">No context files uploaded yet</div>
                    `;
                    askMeFileList.appendChild(emptyState);
                    return;
                }
                
                askMeFiles.forEach((file, index) => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'ask-me-file-item-inline';
                    
                    fileItem.innerHTML = `
                        <span class="file-name">📄 ${file.filename}</span>
                        <button class="btn danger small" onclick="removeAskMeFile(${index})">🗑️</button>
                    `;
                    askMeFileList.appendChild(fileItem);
                });
            }
            
            function toggleFileContent(index) {
                const fileItem = document.querySelectorAll('.ask-me-file-item')[index];
                const contentDiv = fileItem.querySelector('.ask-me-file-content');
                const file = askMeFiles[index];
                
                if (contentDiv.classList.contains('expanded')) {
                    // Collapse
                    const contentPreview = file.content.length > 200 ? 
                        file.content.substring(0, 200) + '...' : file.content;
                    contentDiv.textContent = contentPreview;
                    contentDiv.classList.remove('expanded');
                } else {
                    // Expand
                    contentDiv.textContent = file.content;
                    contentDiv.classList.add('expanded');
                }
            }
            
            async function removeAskMeFile(index) {
                askMeFiles.splice(index, 1);
                displayAskMeFiles();
                updateAdditionalContext();
                
                // If we have a conversation ID and files, update the cache
                if (currentConversationId && askMeFiles.length > 0) {
                    try {
                        // Recreate cache with remaining files
                        const formData = new FormData();
                        formData.append('conversation_id', currentConversationId);
                        
                        // We would need to re-upload the remaining files to update the cache
                        // For now, we'll just log that the cache needs updating
                        console.log('Cache needs updating after file removal');
                    } catch (error) {
                        console.error('Error updating cache after file removal:', error);
                    }
                } else if (currentConversationId && askMeFiles.length === 0) {
                    // If no files left, delete the cache
                    try {
                        await fetch('/api/ask-me-delete-context', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json; charset=utf-8',
                            },
                            body: JSON.stringify({
                                conversation_id: currentConversationId
                            })
                        });
                        currentConversationId = null;
                        console.log('Deleted empty context cache');
                    } catch (error) {
                        console.error('Error deleting empty context cache:', error);
                    }
                }
            }
            
            function updateAdditionalContext() {
                additionalContext = askMeFiles.map(file => file.content).join('\\n');
            }
         
         async function loadStatementsData(filename) {
             try {
                 const response = await fetch('/api/statements-data', {
                     method: 'POST',
                     headers: {
                         'Content-Type': 'application/json; charset=utf-8'
                     },
                     body: JSON.stringify({ filename: filename })
                 });
                 
                 if (!response.ok) {
                     throw new Error('Failed to load statements data');
                 }
                 
                 const data = await response.json();
                 statementsData = data.statements || [];
                 selectedStatements = statementsData.map((_, index) => index); // Pre-select all
                 displayStatements();
                 updateStartButtonState();
             } catch (error) {
                 console.error('Error loading statements data:', error);
                 alert('Failed to load statements data');
             }
         }
         
                             function displayStatements() {
               const statementsList = document.getElementById('statementsList');
               const statementsFileInfo = document.getElementById('statementsFileInfo');
               const statementsSelectAll = document.getElementById('statementsSelectAll');
               
               if (!statementsData || statementsData.length === 0) {
                   statementsList.style.display = 'none';
                   statementsSelectAll.style.display = 'none';
                   return;
               }
               
               statementsList.style.display = 'block';
               statementsSelectAll.style.display = 'block';
               statementsFileInfo.style.display = 'none';
               
               // Update Select All checkbox state
               const selectAllCheckbox = document.getElementById('selectAllStatements');
               if (selectAllCheckbox) {
                   selectAllCheckbox.checked = selectedStatements.length === statementsData.length;
               }
               
               let html = '';
               statementsData.forEach((statement, index) => {
                   const isSelected = selectedStatements.includes(index);
                   
                   html += `
                       <div class="statement-item" onclick="toggleStatementExpansion(${index})">
                           <input type="checkbox" class="statement-checkbox" 
                                  id="statement-checkbox-${index}"
                                  name="statement-checkbox-${index}"
                                  ${isSelected ? 'checked' : ''} 
                                  data-index="${index}"
                                  onclick="event.stopPropagation(); toggleStatementSelection(${index})">
                           <div class="statement-content">
                               <div class="statement-header">
                                   <span class="statement-par">${statement.par}</span>
                                   <div class="statement-text" id="statement-text-${index}">${statement.content}</div>
                               </div>
                           </div>
                       </div>
                   `;
               });
               
               statementsList.innerHTML = html;
           }
         
                   function toggleStatementSelection(index) {
              if (selectedStatements.includes(index)) {
                  selectedStatements = selectedStatements.filter(i => i !== index);
              } else {
                  selectedStatements.push(index);
              }
              
              // Update Select All checkbox state
              const selectAllCheckbox = document.getElementById('selectAllStatements');
              if (selectAllCheckbox) {
                  selectAllCheckbox.checked = selectedStatements.length === statementsData.length;
              }
              
              updateStartButtonState();
          }
         
                   function toggleStatementExpansion(index) {
              const textElement = document.getElementById(`statement-text-${index}`);
              const statement = statementsData[index];
              
              if (textElement.classList.contains('expanded')) {
                  // Collapse
                  textElement.classList.remove('expanded');
              } else {
                  // Expand
                  textElement.classList.add('expanded');
              }
          }
        
        async function uploadKnowledgeFiles(files) {
            for (let file of files) {
                await uploadFile(file, 'knowledge');
            }
        }
        
                 async function uploadStatementsFile(file) {
             await uploadFile(file, 'statements');
             // Load and display statements after upload
             await loadStatementsData(file.name);
         }
        
        async function uploadFile(file, type) {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('type', type);
            
            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('Upload failed');
                }
                
                const result = await response.json();
                console.log(file.name + ' uploaded successfully as ' + type);
                
                // Update the stored filename with the sanitized name from server
                if (type === 'knowledge') {
                    const fileIndex = knowledgeFiles.findIndex(f => f.name === file.name);
                    if (fileIndex !== -1) {
                        knowledgeFiles[fileIndex].name = result.filename;
                        updateKnowledgeFileList();
                    }
                } else if (type === 'statements') {
                    if (statementsFile && statementsFile.name === file.name) {
                        statementsFile.name = result.filename;
                        updateStatementsFileInfo();
                    }
                }
                
            } catch (error) {
                console.error('Failed to upload ' + file.name + ':', error);
            }
        }
        
                 async function startProcessing() {
             if (!statementsFile) {
                 alert('Please upload a statements file');
                 return;
             }
             
             const hasKnowledge = knowledgeFiles.length > 0 || selectedCaches.length > 0;
             if (!hasKnowledge) {
                 alert('Please upload knowledge files or select existing caches');
                 return;
             }
             
             if (selectedStatements.length === 0) {
                 alert('Please select at least one statement to verify');
                 return;
             }
             
             // Show processing status and change button
             const startBtn = document.getElementById('startBtn');
             const processingNote = document.getElementById('processingNote');
             const completionNote = document.getElementById('completionNote');
             
             // Change button to processing state
             startBtn.disabled = true;
             startBtn.className = 'btn processing large';
             startBtn.innerHTML = '⏳ Work in Progress...';
             
             // Show processing note
             processingNote.style.display = 'block';
             completionNote.style.display = 'none';
             
             try {
                 const response = await fetch('/api/process', {
                     method: 'POST',
                     headers: {
                         'Content-Type': 'application/json; charset=utf-8'
                     },
                     body: JSON.stringify({
                         selected_caches: selectedCaches,
                         statements_file: statementsFile.name,
                         knowledge_files: knowledgeFiles.map(f => f.name),
                         selected_statements: selectedStatements
                     })
                 });
                 
                 if (!response.ok) {
                     throw new Error('Failed to start processing');
                 }
                 
                 // Start status polling
                 startStatusPolling();
                 
             } catch (error) {
                 console.error('Error starting processing:', error);
                 alert('Failed to start processing');
                 
                 // Reset button and hide notes on error
                 startBtn.disabled = false;
                 startBtn.className = 'btn success large';
                 startBtn.innerHTML = ' Start Verification Process';
                 processingNote.style.display = 'none';
                 completionNote.style.display = 'none';
             }
         }
        
        function startStatusPolling() {
            statusPolling = setInterval(async () => {
                try {
                    const response = await fetch('/api/status');
                    const status = await response.json();
                    
                    if (status.status === 'completed' || status.status === 'error') {
                        stopStatusPolling();
                        
                        // Update button and notes based on completion status
                        const startBtn = document.getElementById('startBtn');
                        const processingNote = document.getElementById('processingNote');
                        const completionNote = document.getElementById('completionNote');
                        
                        if (status.status === 'completed') {
                            // Change to completed state
                            startBtn.className = 'btn completed large';
                            startBtn.innerHTML = '✅ Verification Completed';
                            startBtn.disabled = true;
                            
                            // Hide processing note and show completion note
                            processingNote.style.display = 'none';
                            completionNote.style.display = 'block';
                        } else {
                            // Reset on error
                            startBtn.disabled = false;
                            startBtn.className = 'btn success large';
                            startBtn.innerHTML = ' Start Verification Process';
                            processingNote.style.display = 'none';
                            completionNote.style.display = 'none';
                        }
                    }
                } catch (error) {
                    console.error('Error polling status:', error);
                }
            }, 1000);
        }
        
        function stopStatusPolling() {
            if (statusPolling) {
                clearInterval(statusPolling);
                statusPolling = null;
            }
        }
        
        function updateStatusDisplay(status) {
            const statusDisplay = document.getElementById('statusDisplay');
            const progressFill = document.getElementById('progressFill');
            const logs = document.getElementById('logs');
            
            if (statusDisplay) {
                statusDisplay.innerHTML = '<div class="status ' + status.status + '"><h3>' + 
                    status.status.charAt(0).toUpperCase() + status.status.slice(1) + '</h3><p><strong>Current Step:</strong> ' + 
                    (status.current_step || 'N/A') + '</p><p><strong>Progress:</strong> ' + 
                    (status.processed_items || 0) + ' / ' + (status.total_items || 0) + '</p><p><strong>Message:</strong> ' + 
                    (status.message || 'N/A') + '</p></div>';
            }
            
            if (progressFill) {
                progressFill.style.width = (status.progress || 0) + '%';
            }
            
            if (logs && status.logs && status.logs.length > 0) {
                const logText = status.logs.join('\\n');
                logs.innerHTML = '<h4>Logs:</h4><pre>' + logText + '</pre>';
            }
        }
        
        function updateDeepAnalysisStatusDisplay(status) {
            const deepStatusDisplay = document.getElementById('deepStatusDisplay');
            const deepProgressFill = document.getElementById('deepProgressFill');
            const deepLogs = document.getElementById('deepLogs');
            
            if (deepStatusDisplay) {
                deepStatusDisplay.innerHTML = '<div class="status ' + status.status + '"><h3>Deep Analysis: ' + 
                    status.status.charAt(0).toUpperCase() + status.status.slice(1) + '</h3><p><strong>Current Step:</strong> ' + 
                    (status.current_step || 'N/A') + '</p><p><strong>Progress:</strong> ' + 
                    (status.processed_items || 0) + ' / ' + (status.total_items || 0) + '</p><p><strong>Message:</strong> ' + 
                    (status.message || 'N/A') + '</p></div>';
            }
            
            if (deepProgressFill) {
                deepProgressFill.style.width = (status.progress || 0) + '%';
            }
            
            if (deepLogs && status.logs && status.logs.length > 0) {
                const logText = status.logs.join('\\n');
                deepLogs.innerHTML = '<h4>Deep Analysis Logs:</h4><pre>' + logText + '</pre>';
            }
        }
        
        async function loadResults() {
            try {
                const response = await fetch('/api/results');
                const results = await response.json();
                displayResults(results);
            } catch (error) {
                console.error('Error loading results:', error);
                alert('Failed to load results');
            }
        }
        
                 function displayResults(results) {
             const resultsTable = document.getElementById('resultsTable');
             
             if (!results || results.length === 0) {
                 resultsTable.innerHTML = '<p>No results found</p>';
                 return;
             }
             
                           let tableHTML = '<table class="results-table"><thead><tr>' +
                  '<th>Par Number</th>' +
                  '<th class="par-content">Par Content</th>' +
                  '<th class="center-align">Accuracy</th>' +
                  '<th class="center-align">Accuracy Rating</th>' +
                  '<th class="center-align">Inaccuracy Type</th>' +
                  '<th class="description">Description</th>' +
                  '</tr></thead><tbody>';
             
                           results.forEach((result, index) => {
                  const degreeOfAccuracy = result['Degree of Accuracy'] || 0;
                  const isLowAccuracy = degreeOfAccuracy < 6;
                  const accuracyClass = isLowAccuracy ? 'low-accuracy' : '';
                  
                  // Truncate par content for display
                  const parContent = result['Par Context'] || '';
                  const shortContent = parContent.length > 200 ? 
                      parContent.substring(0, 200) + '...' : parContent;
                  
                  tableHTML += '<tr>' +
                      '<td>' + (result['Analysed Par number'] || '') + '</td>' +
                      '<td class="par-content" onclick="toggleParContent(' + index + ')">' +
                      '<div class="par-content-text" id="par-content-' + index + '" data-full-content="' + parContent.replace(/"/g, '&quot;') + '">' + shortContent + '</div>' +
                      '</td>' +
                      '<td class="center-align">' + (result['Is Accurate'] || '') + '</td>' +
                      '<td class="center-align ' + accuracyClass + '">' + degreeOfAccuracy + '</td>' +
                      '<td class="center-align">' + (result['Inaccuracy Type'] || '') + '</td>' +
                      '<td class="description">' + (result['Description'] || '') + '</td>' +
                      '</tr>';
              });
             
             tableHTML += '</tbody></table>';
             resultsTable.innerHTML = tableHTML;
         }
         
         function toggleParContent(index) {
             const contentElement = document.getElementById('par-content-' + index);
             if (contentElement.classList.contains('expanded')) {
                 // Collapse
                 const fullContent = contentElement.getAttribute('data-full-content');
                 const shortContent = fullContent.length > 200 ? 
                     fullContent.substring(0, 200) + '...' : fullContent;
                 contentElement.textContent = shortContent;
                 contentElement.classList.remove('expanded');
             } else {
                 // Expand
                 const fullContent = contentElement.getAttribute('data-full-content');
                 if (fullContent) {
                     contentElement.textContent = fullContent;
                     contentElement.classList.add('expanded');
                 }
             }
         }
        
        async function downloadResults() {
            try {
                const response = await fetch('/api/results/download');
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'verification_results.xlsx';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Error downloading results:', error);
                alert('Failed to download results');
            }
        }
        
        async function startDeepAnalysis() {
            // Show warning dialog
            const confirmed = confirm('⚠️ WARNING: Deep analysis is a slow and expensive search and should be done on a limited set of data!\\n\\nDo you want to continue with the deep analysis?');
            
            if (!confirmed) {
                return;
            }
            
            const deepAnalysisBtn = document.getElementById('deepAnalysisBtn');
            const processingNote = document.getElementById('processingNote');
            const completionNote = document.getElementById('completionNote');
            
            // Disable button and show processing state
            deepAnalysisBtn.disabled = true;
            deepAnalysisBtn.className = 'btn processing large';
            deepAnalysisBtn.innerHTML = '🔍 Deep Analysis Running...';
            processingNote.style.display = 'block';
            completionNote.style.display = 'none';
            
            try {
                const response = await fetch('/api/deep-analysis', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8'
                    },
                    body: JSON.stringify({
                        selected_caches: selectedCaches,
                        statements_file: statementsFile.name,
                        knowledge_files: knowledgeFiles.map(f => f.name),
                        selected_statements: selectedStatements
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to start deep analysis');
                }
                
                // Start status polling for deep analysis
                startDeepAnalysisStatusPolling();
                
            } catch (error) {
                console.error('Error starting deep analysis:', error);
                alert('Failed to start deep analysis');
                
                // Reset button and hide notes on error
                deepAnalysisBtn.disabled = false;
                deepAnalysisBtn.className = 'btn warning large';
                deepAnalysisBtn.innerHTML = '🔍 Start Deep Analysis';
                processingNote.style.display = 'none';
                completionNote.style.display = 'none';
            }
        }
        
        function startDeepAnalysisStatusPolling() {
            deepAnalysisStatusPolling = setInterval(async () => {
                try {
                    const response = await fetch('/api/deep-analysis-status');
                    const status = await response.json();
                    
                    // Update status display
                    updateDeepAnalysisStatusDisplay(status);
                    
                    if (status.status === 'completed' || status.status === 'error') {
                        stopDeepAnalysisStatusPolling();
                        
                        // Update button and notes based on completion status
                        const deepAnalysisBtn = document.getElementById('deepAnalysisBtn');
                        const processingNote = document.getElementById('processingNote');
                        const completionNote = document.getElementById('completionNote');
                        
                        if (status.status === 'completed') {
                            // Change to completed state but keep warning color
                            deepAnalysisBtn.className = 'btn warning large';
                            deepAnalysisBtn.innerHTML = '✅ Deep Analysis Completed';
                            deepAnalysisBtn.disabled = true;
                            
                            // Hide processing note and show completion note
                            processingNote.style.display = 'none';
                            completionNote.style.display = 'block';
                        } else {
                            // Reset on error
                            deepAnalysisBtn.disabled = false;
                            deepAnalysisBtn.className = 'btn warning large';
                            deepAnalysisBtn.innerHTML = '🔍 Start Deep Analysis';
                            processingNote.style.display = 'none';
                            completionNote.style.display = 'none';
                        }
                    }
                } catch (error) {
                    console.error('Error polling deep analysis status:', error);
                }
            }, 1000);
        }
        
        function stopDeepAnalysisStatusPolling() {
            if (deepAnalysisStatusPolling) {
                clearInterval(deepAnalysisStatusPolling);
                deepAnalysisStatusPolling = null;
            }
        }
        
        async function loadDeepResults() {
            try {
                const response = await fetch('/api/deep-results');
                const results = await response.json();
                displayDeepResults(results);
            } catch (error) {
                console.error('Error loading deep results:', error);
                alert('Failed to load deep results');
            }
        }
        
        function displayDeepResults(results) {
            const resultsTable = document.getElementById('resultsTable');
            
            if (!results || results.length === 0) {
                resultsTable.innerHTML = '<p>No deep analysis results found</p>';
                return;
            }
            
            let tableHTML = '<table class="results-table"><thead><tr>' +
                '<th>Par Number</th>' +
                '<th class="par-content">Par Content</th>' +
                '<th class="center-align">Legal Analysis</th>' +
                '<th class="center-align">Appeal Grounds</th>' +
                '<th class="center-align">Degree of Accuracy</th>' +
                '</tr></thead><tbody>';
            
            results.forEach((result, index) => {
                // Truncate par content for display
                const parContent = result['Par Context'] || '';
                const shortContent = parContent.length > 200 ? 
                    parContent.substring(0, 200) + '...' : parContent;
                
                // Get degree of accuracy and apply styling
                const degreeOfAccuracy = result['Degree of Accuracy'] || 0;
                const isLowAccuracy = degreeOfAccuracy < 6;
                const accuracyClass = isLowAccuracy ? 'low-accuracy' : '';
                
                tableHTML += '<tr>' +
                    '<td>' + (result['Analysed Par number'] || '') + '</td>' +
                    '<td class="par-content" onclick="toggleParContent(' + index + ')">' +
                    '<div class="par-content-text" id="par-content-' + index + '" data-full-content="' + parContent.replace(/"/g, '&quot;') + '">' + shortContent + '</div>' +
                    '</td>' +
                    '<td class="center-align">' + (result['Legal Analysis'] || '') + '</td>' +
                    '<td class="center-align">' + (result['Appeal Grounds'] || '') + '</td>' +
                    '<td class="center-align ' + accuracyClass + '">' + degreeOfAccuracy + '</td>' +
                    '</tr>';
            });
            
            tableHTML += '</tbody></table>';
            resultsTable.innerHTML = tableHTML;
        }
        
        async function downloadDeepResults() {
            try {
                const response = await fetch('/api/deep-results/download');
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'Deep Verification results.xlsx';
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                window.URL.revokeObjectURL(url);
            } catch (error) {
                console.error('Error downloading deep results:', error);
                alert('Failed to download deep results');
            }
        }
        
        async function loadCaches() {
            try {
                // Ensure custom categories are loaded before displaying caches
                if (!window.customCategories) {
                    loadCustomCategories();
                }
                
                // Load categories from server
                try {
                    const categoriesResponse = await fetch('/api/cache-categories');
                    if (categoriesResponse.ok) {
                        const categoriesData = await categoriesResponse.json();
                        window.cacheCategories = categoriesData.categories || {};
                        console.log('Loaded categories from server:', window.cacheCategories);
                    }
                } catch (error) {
                    console.warn('Failed to load categories from server, using localStorage:', error);
                    // Fallback to localStorage
                    const savedCategories = localStorage.getItem('veridoc_cache_categories');
                    if (savedCategories) {
                        window.cacheCategories = JSON.parse(savedCategories);
                    } else {
                        window.cacheCategories = {};
                    }
                }
                
                // Sync localStorage categories to server if they exist
                const savedCategories = localStorage.getItem('veridoc_cache_categories');
                if (savedCategories) {
                    const localCategories = JSON.parse(savedCategories);
                    if (Object.keys(localCategories).length > 0) {
                        try {
                            console.log('Syncing localStorage categories to server:', localCategories);
                            const syncResponse = await fetch('/api/cache-categories', {
                                method: 'POST',
                                headers: {
                                    'Content-Type': 'application/json; charset=utf-8'
                                },
                                body: JSON.stringify({
                                    categories: localCategories
                                })
                            });
                            
                            if (syncResponse.ok) {
                                console.log('Successfully synced localStorage categories to server');
                                // Update the server categories in memory
                                const updatedCategories = await syncResponse.json();
                                window.cacheCategories = updatedCategories.categories || window.cacheCategories;
                            } else {
                                console.warn('Failed to sync localStorage categories to server - response not ok');
                            }
                        } catch (error) {
                            console.warn('Failed to sync localStorage categories to server:', error);
                        }
                    }
                }
                
                const response = await fetch('/api/caches');
                const caches = await response.json();
                displayCaches(caches);
            } catch (error) {
                console.error('Error loading caches:', error);
                alert('Failed to load caches: ' + error.message);
            }
        }
        
                 function displayCaches(caches) {
             const cachesTable = document.getElementById('cachesTable');
             
             if (!caches || caches.length === 0) {
                 cachesTable.innerHTML = '<p><em>No cached files found</em></p>';
                 return;
             }
             
                           // Initialize cache names and categories if not exists
              if (!window.cacheNames) {
                  window.cacheNames = {};
              }
              if (!window.cacheCategories) {
                  window.cacheCategories = {};
              }
             
                           // Group caches by category using server-side categories
             const categories = {};
             
             // Initialize with all available categories
             if (window.allCategories) {
                 window.allCategories.forEach(category => {
                     categories[category] = [];
                 });
             } else {
                 // Fallback to default categories
                 categories['Uncategorized'] = [];
                 categories['A - Principal case documents'] = [];
                 categories['AA - Trial Documents'] = [];
                 categories['B - Factual witness statements'] = [];
                 categories['C - Law expert reports'] = [];
                 categories['D - Forensic and valuation reports'] = [];
                 categories['Hearing Transcripts'] = [];
                 categories['Orders & Judgements'] = [];
                 categories['Other'] = [];
             }
             
             caches.forEach(cache => {
                 // Try to auto-categorize based on cache name if no category is set
                 let category = window.cacheCategories[cache.cache_id];
                 if (!category) {
                     category = autoCategorizeCache(cache.original_name);
                     if (category !== 'Uncategorized') {
                         window.cacheCategories[cache.cache_id] = category;
                         localStorage.setItem('veridoc_cache_categories', JSON.stringify(window.cacheCategories));
                     }
                 }
                 
                 if (categories[category]) {
                     categories[category].push(cache);
                 } else {
                     categories['Other'].push(cache);
                 }
             });
             
             let html = '';
             
             // Create category sections
             Object.keys(categories).forEach(categoryName => {
                 const categoryCaches = categories[categoryName];
                 if (categoryCaches.length === 0) return;
                 
                 const isUncategorized = categoryName === 'Uncategorized';
                 const categoryClass = isUncategorized ? 'uncategorized' : '';
                 
                                   html += `
                      <div class="cache-category ${categoryClass}">
                          <div class="category-header">
                              <span class="category-toggle" id="toggle-${categoryName.replace(/[^a-zA-Z0-9]/g, '')}" onclick="toggleCategory('${categoryName}')">+</span>
                              <span onclick="toggleCategory('${categoryName}')" class="flex-cursor">${categoryName} (${categoryCaches.length})</span>
                              <label class="flex-label-small">
                                                                      <input type="checkbox" class="category-select-all checkbox-margin" 
                                           id="category-select-all-${categoryName.replace(/[^a-zA-Z0-9]/g, '')}"
                                           name="category-select-all-${categoryName.replace(/[^a-zA-Z0-9]/g, '')}"
                                           data-category="${categoryName}" aria-label="Select all items in category: ${categoryName}" title="Select all items in category: ${categoryName}">
                                  Select All
                              </label>
                          </div>
                          <div class="category-content" id="content-${categoryName.replace(/[^a-zA-Z0-9]/g, '')}">
                  `;
                 
                                   categoryCaches.forEach(cache => {
                      // Use saved name if available, otherwise use original name
                      const displayName = window.cacheNames[cache.cache_id] || cache.original_name;
                      
                      html += `
                          <div class="cache-item">
                              <input type="checkbox" class="cache-checkbox" 
                                     id="cache-checkbox-${cache.cache_id}"
                                     name="cache-checkbox-${cache.cache_id}"
                                     value="${cache.cache_id}" checked aria-label="Select cache item: ${displayName}" title="Select cache item: ${displayName}">
                              <div class="cache-name">
                                  <textarea class="cache-name-editable" 
                                         id="cache-name-${cache.cache_id}"
                                         name="cache-name-${cache.cache_id}"
                                         data-cache-id="${cache.cache_id}"
                                         onblur="updateCacheName('${cache.cache_id}', this.value)"
                                         onkeypress="if(event.key==='Enter' && !event.shiftKey) { event.preventDefault(); this.blur(); }">${displayName}</textarea>
                              </div>
                              <div class="cache-size">${cache.total_size_mb.toFixed(1)} MB</div>
                              <select class="cache-category-select" 
                                      id="category-select-${cache.cache_id}"
                                      name="category-select-${cache.cache_id}"
                                      onchange="handleCategoryChange('${cache.cache_id}', this.value)"
                                      aria-label="Select category for ${displayName}"
                                      title="Select category for ${displayName}">
                                  ${generateCategoryOptions(cache.cache_id)}
                              </select>
                          </div>
                      `;
                  });
                 
                 html += `
                         </div>
                     </div>
                 `;
             });
             
             cachesTable.innerHTML = html;
             
             // Pre-select all caches
             selectedCaches = caches.map(cache => cache.cache_id);
             
                                          // Add event listeners for checkboxes
               const cacheCheckboxes = document.querySelectorAll('.cache-checkbox');
               cacheCheckboxes.forEach(cb => {
                   cb.addEventListener('change', function() {
                       if (this.checked) {
                           if (!selectedCaches.includes(this.value)) {
                               selectedCaches.push(this.value);
                           }
                       } else {
                           selectedCaches = selectedCaches.filter(id => id !== this.value);
                       }
                       updateStartButtonState();
                       updateCategorySelectAllStates();
                   });
               });
              
                                                           // Add event listeners for category select all checkboxes
                const categorySelectAllCheckboxes = document.querySelectorAll('.category-select-all');
                categorySelectAllCheckboxes.forEach(cb => {
                    cb.addEventListener('change', function() {
                        const category = this.getAttribute('data-category');
                        const categoryElement = this.closest('.cache-category');
                        const categoryContent = categoryElement.querySelector('.category-content');
                        const checkboxes = categoryContent.querySelectorAll('.cache-checkbox');
                        
                        checkboxes.forEach(checkbox => {
                            checkbox.checked = this.checked;
                            const cacheId = checkbox.value;
                            
                            if (this.checked) {
                                if (!selectedCaches.includes(cacheId)) {
                                    selectedCaches.push(cacheId);
                                }
                            } else {
                                selectedCaches = selectedCaches.filter(id => id !== cacheId);
                            }
                        });
                        
                        updateStartButtonState();
                        updateCategorySelectAllStates();
                    });
                });
             
                           // Update start button state after pre-selecting
              updateStartButtonState();
              
                             // Initialize auto-resize for all cache name textareas
               const cacheNameTextareas = document.querySelectorAll('.cache-name-editable');
               cacheNameTextareas.forEach(textarea => {
                   // Set initial height based on content
                   textarea.style.height = 'auto';
                   textarea.style.height = textarea.scrollHeight + 'px';
               });
               
               // Update category select all states after initial load
               updateCategorySelectAllStates();
          }
         
         function toggleCategory(categoryName) {
             const toggleId = 'toggle-' + categoryName.replace(/[^a-zA-Z0-9]/g, '');
             const contentId = 'content-' + categoryName.replace(/[^a-zA-Z0-9]/g, '');
             
             const toggle = document.getElementById(toggleId);
             const content = document.getElementById(contentId);
             
             if (content.classList.contains('expanded')) {
                 content.classList.remove('expanded');
                 toggle.classList.remove('expanded');
             } else {
                 content.classList.add('expanded');
                 toggle.classList.add('expanded');
             }
         }
         
                   function updateCacheName(cacheId, newName) {
              // Save to localStorage for persistence
              if (!window.cacheNames) {
                  window.cacheNames = {};
              }
              window.cacheNames[cacheId] = newName;
              localStorage.setItem('veridoc_cache_names', JSON.stringify(window.cacheNames));
              
              console.log(`Updated cache ${cacheId} name to: ${newName}`);
          }
         
                                                function handleCategoryChange(cacheId, selectedValue) {
             if (selectedValue === '__ADD_NEW__') {
                 // Show dialog to add new category
                 addNewCategory(cacheId);
             } else if (selectedValue === '__REMOVE_CATEGORY__') {
                 // Show dialog to remove category
                 removeCategory(cacheId);
             } else {
                 // Update category normally
                 updateCacheCategory(cacheId, selectedValue);
             }
         }
         
             function addNewCategory(cacheId) {
        const newCategoryName = prompt(`Enter the name for the new category:

Examples: "E - Financial Documents", "F - Correspondence", "G - Evidence"`);
        if (newCategoryName && newCategoryName.trim()) {
                 const trimmedName = newCategoryName.trim();
                 
                 // Check if category already exists
                 const defaultCategories = [
                     'Uncategorized',
                     'A - Principal case documents',
                     'AA - Trial Documents',
                     'B - Factual witness statements',
                     'C - Law expert reports',
                     'D - Forensic and valuation reports',
                     'Hearing Transcripts',
                     'Orders & Judgements',
                     'Other'
                 ];
                 
                 if (defaultCategories.includes(trimmedName)) {
                     alert('This category already exists as a default category.');
                     return;
                 }
                 
                 if (window.customCategories && window.customCategories.includes(trimmedName)) {
                     alert('This category already exists.');
                     return;
                 }
                 
                 // Add the new category to all dropdowns
                 addCategoryToAllDropdowns(trimmedName);
                 
                 // Update the cache category
                 updateCacheCategory(cacheId, trimmedName);
                 
                 // Reset the dropdown to show the new category as selected
                 setTimeout(() => {
                     const select = document.querySelector(`[data-cache-id="${cacheId}"]`).closest('.cache-item').querySelector('.cache-category-select');
                     if (select) {
                         select.value = trimmedName;
                     }
                 }, 100);
             } else {
                 // Reset the dropdown to the previous value if user cancels
                 setTimeout(() => {
                     const select = document.querySelector(`[data-cache-id="${cacheId}"]`).closest('.cache-item').querySelector('.cache-category-select');
                     if (select) {
                         const currentCategory = window.cacheCategories[cacheId] || 'Uncategorized';
                         select.value = currentCategory;
                     }
                 }, 100);
             }
         }
         
         function removeCategory(cacheId) {
             // Get all custom categories
             if (!window.customCategories || window.customCategories.length === 0) {
                 alert('No custom categories available to remove.');
                 // Reset the dropdown to the previous value
                 setTimeout(() => {
                     const select = document.querySelector(`[data-cache-id="${cacheId}"]`).closest('.cache-item').querySelector('.cache-category-select');
                     if (select) {
                         const currentCategory = window.cacheCategories[cacheId] || 'Uncategorized';
                         select.value = currentCategory;
                     }
                 }, 100);
                 return;
             }
             
             // Create a list of custom categories for selection
             const customCategoriesList = window.customCategories.map(category => `"${category}"`).join('\\n');
             
             const categoryToRemove = prompt(`Select a custom category to remove:

Available custom categories:
${customCategoriesList}

Enter the exact category name to remove:`);
             
             if (categoryToRemove && categoryToRemove.trim()) {
                 const trimmedName = categoryToRemove.trim();
                 
                 // Check if the category exists
                 if (!window.customCategories.includes(trimmedName)) {
                     alert(`Category "${trimmedName}" not found in custom categories.`);
                     // Reset the dropdown to the previous value
                     setTimeout(() => {
                         const select = document.querySelector(`[data-cache-id="${cacheId}"]`).closest('.cache-item').querySelector('.cache-category-select');
                         if (select) {
                             const currentCategory = window.cacheCategories[cacheId] || 'Uncategorized';
                             select.value = currentCategory;
                         }
                     }, 100);
                     return;
                 }
                 
                 // Check if any documents are currently assigned to this category
                 const documentsInCategory = [];
                 if (window.cacheCategories) {
                     for (const [docId, category] of Object.entries(window.cacheCategories)) {
                         if (category === trimmedName) {
                             documentsInCategory.push(docId);
                         }
                     }
                 }
                 
                 if (documentsInCategory.length > 0) {
                     const confirmMessage = `Category "${trimmedName}" has ${documentsInCategory.length} document(s) assigned to it. 
                     
These documents will be moved to "Uncategorized" when the category is removed.
                     
Do you want to continue?`;
                     
                     if (!confirm(confirmMessage)) {
                         // Reset the dropdown to the previous value
                         setTimeout(() => {
                             const select = document.querySelector(`[data-cache-id="${cacheId}"]`).closest('.cache-item').querySelector('.cache-category-select');
                             if (select) {
                                 const currentCategory = window.cacheCategories[cacheId] || 'Uncategorized';
                                 select.value = currentCategory;
                             }
                         }, 100);
                         return;
                     }
                     
                     // Move all documents in this category to "Uncategorized"
                     documentsInCategory.forEach(docId => {
                         updateCacheCategory(docId, 'Uncategorized');
                     });
                 }
                 
                 // Remove the category from custom categories
                 window.customCategories = window.customCategories.filter(cat => cat !== trimmedName);
                 localStorage.setItem('veridoc_custom_categories', JSON.stringify(window.customCategories));
                 
                 // Remove the category section from the UI
                 const categoryElements = document.querySelectorAll('.cache-category');
                 for (const categoryElement of categoryElements) {
                     const categoryName = categoryElement.querySelector('.category-header span:nth-child(2)').textContent.split(' (')[0];
                     if (categoryName === trimmedName) {
                         categoryElement.remove();
                         break;
                     }
                 }
                 
                 // Refresh all dropdowns to remove the deleted category
                 refreshAllCategoryDropdowns();
                 
                 alert(`Category "${trimmedName}" has been removed successfully.`);
             } else {
                 // Reset the dropdown to the previous value if user cancels
                 setTimeout(() => {
                     const select = document.querySelector(`[data-cache-id="${cacheId}"]`).closest('.cache-item').querySelector('.cache-category-select');
                     if (select) {
                         const currentCategory = window.cacheCategories[cacheId] || 'Uncategorized';
                         select.value = currentCategory;
                     }
                 }, 100);
             }
         }
         
         function addCategoryToAllDropdowns(newCategoryName) {
             // Store the new category in localStorage for persistence
             if (!window.customCategories) {
                 window.customCategories = [];
             }
             if (!window.customCategories.includes(newCategoryName)) {
                 window.customCategories.push(newCategoryName);
                 localStorage.setItem('veridoc_custom_categories', JSON.stringify(window.customCategories));
             }
             
             // Refresh all dropdowns to include the new category
             refreshAllCategoryDropdowns();
         }
         
         function refreshAllCategoryDropdowns() {
             // Get all category dropdowns
             const categoryDropdowns = document.querySelectorAll('.cache-category-select');
             
             categoryDropdowns.forEach(select => {
                 const currentValue = select.value;
                 const cacheId = select.closest('.cache-item').querySelector('[data-cache-id]').getAttribute('data-cache-id');
                 
                 // Generate new options
                 const newOptions = generateCategoryOptions(cacheId);
                 
                 // Replace the select content
                 select.innerHTML = newOptions;
                 
                 // Restore the selected value
                 select.value = currentValue;
             });
         }
         
         function loadCustomCategories() {
             // Load custom categories from localStorage
             const savedCategories = localStorage.getItem('veridoc_custom_categories');
             if (savedCategories) {
                 window.customCategories = JSON.parse(savedCategories);
             } else {
                 window.customCategories = [];
             }
         }
         
         function generateCategoryOptions(cacheId) {
             // Ensure custom categories are loaded
             if (!window.customCategories) {
                 loadCustomCategories();
             }
             
             const currentCategory = window.cacheCategories[cacheId] || 'Uncategorized';
             let options = '';
             
             // Default categories
             const defaultCategories = [
                 'Uncategorized',
                 'A - Principal case documents',
                 'AA - Trial Documents',
                 'B - Factual witness statements',
                 'C - Law expert reports',
                 'D - Forensic and valuation reports',
                 'Hearing Transcripts',
                 'Orders & Judgements',
                 'Other'
             ];
             
             // Add default categories
             defaultCategories.forEach(category => {
                 const selected = currentCategory === category ? 'selected' : '';
                 options += `<option value="${category}" ${selected}>${category}</option>`;
             });
             
             // Add custom categories
             if (window.customCategories && window.customCategories.length > 0) {
                 window.customCategories.forEach(category => {
                     const selected = currentCategory === category ? 'selected' : '';
                     options += `<option value="${category}" ${selected}>${category}</option>`;
                 });
             }
             
             // Add "Add New Category" option
             options += `<option value="__ADD_NEW__">➕ Add New Category...</option>`;
             
             // Add "Remove Category" option if there are custom categories
             if (window.customCategories && window.customCategories.length > 0) {
                 options += `<option value="__REMOVE_CATEGORY__">🗑️ Remove Category...</option>`;
             }
             
             return options;
          }
         
          async function syncCategoriesToServer() {
              try {
                  console.log('Starting category synchronization...');
                  
                  // Get categories from localStorage
                  const savedCategories = localStorage.getItem('veridoc_cache_categories');
                  if (!savedCategories) {
                      alert('No categories found in localStorage to sync.');
                      return;
                  }
                  
                  const localCategories = JSON.parse(savedCategories);
                  console.log('Categories from localStorage:', localCategories);
                  
                  if (Object.keys(localCategories).length === 0) {
                      alert('No categories found in localStorage to sync.');
                      return;
                  }
                  
                  // Sync to server
                  const response = await fetch('/api/cache-categories', {
                      method: 'POST',
                      headers: {
                          'Content-Type': 'application/json; charset=utf-8'
                      },
                      body: JSON.stringify({
                          categories: localCategories
                      })
                  });
                  
                  if (response.ok) {
                      const result = await response.json();
                      console.log('Successfully synced categories to server:', result);
                      alert(`Successfully synced ${Object.keys(localCategories).length} category assignments to server.`);
                      
                      // Update the server categories in memory
                      window.cacheCategories = result.categories || window.cacheCategories;
                  } else {
                      console.error('Failed to sync categories to server');
                      alert('Failed to sync categories to server. Please try again.');
                  }
              } catch (error) {
                  console.error('Error syncing categories to server:', error);
                  alert('Error syncing categories to server: ' + error.message);
              }
          }
         
                                       async function updateCacheCategory(cacheId, newCategory) {
               // Store category in memory and server-side storage for persistence
               if (!window.cacheCategories) {
                   window.cacheCategories = {};
               }
               window.cacheCategories[cacheId] = newCategory;
               
               // Save to server
               try {
                   const response = await fetch('/api/cache-categories', {
                       method: 'POST',
                       headers: {
                           'Content-Type': 'application/json; charset=utf-8'
                       },
                       body: JSON.stringify({
                           categories: { [cacheId]: newCategory }
                       })
                   });
                   
                   if (!response.ok) {
                       console.error('Failed to save category to server');
                   }
               } catch (error) {
                   console.error('Error saving category to server:', error);
               }
               
               // Also save to localStorage as backup
               localStorage.setItem('veridoc_cache_categories', JSON.stringify(window.cacheCategories));
               
               console.log(`Updated cache ${cacheId} category to: ${newCategory}`);
               
               // Instead of refreshing the entire display, just move the cache item to the new category
               moveCacheToCategory(cacheId, newCategory);
           }
           
           function moveCacheToCategory(cacheId, newCategory) {
               // Find the cache item element
               const cacheItem = document.querySelector(`[data-cache-id="${cacheId}"]`).closest('.cache-item');
               if (!cacheItem) return;
               
               // Find the target category content
               let targetCategoryContent = null;
               const categoryElements = document.querySelectorAll('.cache-category');
               
               for (const categoryElement of categoryElements) {
                   const categoryName = categoryElement.querySelector('.category-header span:nth-child(2)').textContent.split(' (')[0];
                   if (categoryName === newCategory) {
                       targetCategoryContent = categoryElement.querySelector('.category-content');
                       break;
                   }
               }
               
               // If target category doesn't exist, create it
               if (!targetCategoryContent) {
                   const cachesTable = document.getElementById('cachesTable');
                   const newCategoryElement = document.createElement('div');
                   newCategoryElement.className = 'cache-category';
                   newCategoryElement.innerHTML = `
                       <div class="category-header">
                           <span class="category-toggle" id="toggle-${newCategory.replace(/[^a-zA-Z0-9]/g, '')}" onclick="toggleCategory('${newCategory}')">+</span>
                           <span onclick="toggleCategory('${newCategory}')" class="flex-cursor">${newCategory} (1)</span>
                           <label class="flex-label-small">
                                                                <input type="checkbox" class="category-select-all checkbox-margin" 
                                        id="category-select-all-${newCategory.replace(/[^a-zA-Z0-9]/g, '')}"
                                        name="category-select-all-${newCategory.replace(/[^a-zA-Z0-9]/g, '')}"
                                        data-category="${newCategory}" checked aria-label="Select all items in category: ${newCategory}" title="Select all items in category: ${newCategory}">
                               Select All
                           </label>
                       </div>
                       <div class="category-content expanded" id="content-${newCategory.replace(/[^a-zA-Z0-9]/g, '')}">
                       </div>
                   `;
                   
                   // Insert the new category in the appropriate position based on category order
                   const categoryOrder = [
                       'A - Principal case documents',
                       'AA - Trial Documents', 
                       'B - Factual witness statements',
                       'C - Law expert reports',
                       'D - Forensic and valuation reports',
                       'Hearing Transcripts',
                       'Orders & Judgements',
                       'Other'
                   ];
                   
                   let inserted = false;
                   for (const existingCategory of categoryElements) {
                       const existingName = existingCategory.querySelector('.category-header span:nth-child(2)').textContent.split(' (')[0];
                       const existingIndex = categoryOrder.indexOf(existingName);
                       const newIndex = categoryOrder.indexOf(newCategory);
                       
                       // If new category is not in the predefined order, place it after "Other" but before "Uncategorized"
                       if (newIndex === -1) {
                           if (existingName === 'Uncategorized') {
                               cachesTable.insertBefore(newCategoryElement, existingCategory);
                               inserted = true;
                               break;
                           }
                       } else if (existingIndex !== -1 && newIndex < existingIndex) {
                           cachesTable.insertBefore(newCategoryElement, existingCategory);
                           inserted = true;
                           break;
                       }
                   }
                   
                   if (!inserted) {
                       cachesTable.appendChild(newCategoryElement);
                   }
                   
                   targetCategoryContent = newCategoryElement.querySelector('.category-content');
                   
                   // Add event listeners for the new category
                   const newCategorySelectAll = newCategoryElement.querySelector('.category-select-all');
                   newCategorySelectAll.addEventListener('change', function() {
                       const category = this.getAttribute('data-category');
                       const categoryElement = this.closest('.cache-category');
                       const categoryContent = categoryElement.querySelector('.category-content');
                       const checkboxes = categoryContent.querySelectorAll('.cache-checkbox');
                       
                       checkboxes.forEach(checkbox => {
                           checkbox.checked = this.checked;
                           const cacheId = checkbox.value;
                           
                           if (this.checked) {
                               if (!selectedCaches.includes(cacheId)) {
                                   selectedCaches.push(cacheId);
                               }
                           } else {
                               selectedCaches = selectedCaches.filter(id => id !== cacheId);
                           }
                       });
                       updateStartButtonState();
                   });
                   
                   // Add toggle functionality
                   const newCategoryToggle = newCategoryElement.querySelector('.category-toggle');
                   newCategoryToggle.addEventListener('click', function(e) {
                       e.stopPropagation();
                       const categoryName = this.parentElement.querySelector('span:nth-child(2)').textContent.split(' (')[0];
                       toggleCategory(categoryName);
                   });
               }
               
               // Move the cache item to the target category
               targetCategoryContent.appendChild(cacheItem);
               
               // Update category counts
               updateCategoryCounts();
               
               // Update category select all states
               updateCategorySelectAllStates();
           }
           
           function updateCategoryCounts() {
               const categoryElements = document.querySelectorAll('.cache-category');
               categoryElements.forEach(categoryElement => {
                   const categoryContent = categoryElement.querySelector('.category-content');
                   const cacheItems = categoryContent.querySelectorAll('.cache-item');
                   const countSpan = categoryElement.querySelector('.category-header span:nth-child(2)');
                   const categoryName = countSpan.textContent.split(' (')[0];
                   countSpan.textContent = `${categoryName} (${cacheItems.length})`;
               });
           }
           
           function updateCategorySelectAllStates() {
               // Get all category elements
               const categoryElements = document.querySelectorAll('.cache-category');
               
               categoryElements.forEach(categoryElement => {
                   const selectAllCheckbox = categoryElement.querySelector('.category-select-all');
                   const cacheCheckboxes = categoryElement.querySelectorAll('.cache-checkbox');
                   
                   if (selectAllCheckbox && cacheCheckboxes.length > 0) {
                       const checkedCount = Array.from(cacheCheckboxes).filter(cb => cb.checked).length;
                       const totalCount = cacheCheckboxes.length;
                       
                       // Remove indeterminate state first
                       selectAllCheckbox.indeterminate = false;
                       
                       if (checkedCount === 0) {
                           // None selected
                           selectAllCheckbox.checked = false;
                       } else if (checkedCount === totalCount) {
                           // All selected
                           selectAllCheckbox.checked = true;
                       } else {
                           // Some selected (indeterminate state)
                           selectAllCheckbox.checked = false;
                           selectAllCheckbox.indeterminate = true;
                       }
                   }
               });
           }
           
           async function indexKnowledgeFiles() {
               if (knowledgeFiles.length === 0) {
                   alert('Please upload knowledge files first');
                   return;
               }
               
               const indexBtn = document.getElementById('indexFilesBtn');
               const originalText = indexBtn.textContent;
               
               try {
                   // Show processing state
                   indexBtn.disabled = true;
                   indexBtn.textContent = '⏳ Indexing...';
                   
                   console.log('Indexing files:', knowledgeFiles.map(f => f.name));
                   
                   const response = await fetch('/api/index-files', {
                       method: 'POST',
                       headers: {
                           'Content-Type': 'application/json; charset=utf-8'
                       },
                       body: JSON.stringify({
                           knowledge_files: knowledgeFiles.map(f => f.name)
                       })
                   });
                   
                   const result = await response.json();
                   
                   if (!response.ok) {
                       throw new Error(result.detail || 'Failed to index files');
                   }
                   
                   // Show detailed results
                   let message = `Successfully indexed ${result.indexed_count} files. New caches have been created.`;
                   if (result.failed_files && result.failed_files.length > 0) {
                       message += `\n\nFailed files:\n${result.failed_files.join('\\n')}`;
                   }
                   
                   alert(message);
                   
                   // Clear uploaded files and reload caches
                   knowledgeFiles = [];
                   updateKnowledgeFileList();
                   await loadCaches();
                   updateStartButtonState();
                   
               } catch (error) {
                   console.error('Error indexing files:', error);
                   alert('Failed to index files: ' + error.message);
               } finally {
                   // Restore button state
                   indexBtn.disabled = false;
                   indexBtn.textContent = originalText;
               }
           }
          
                     function autoResizeTextarea(element) {
               element.style.height = 'auto';
               element.style.height = element.scrollHeight + 'px';
           }
           
           // Add event listener for textarea auto-resize
           document.addEventListener('input', function(e) {
               if (e.target.classList.contains('cache-name-editable')) {
                   autoResizeTextarea(e.target);
               }
           });
        
        async function clearAllCategories() {
            if (confirm('Are you sure you want to clear all cache categories? This will reset all caches to "Uncategorized".')) {
                window.cacheCategories = {};
                
                // Clear from server
                try {
                    const response = await fetch('/api/cache-categories', {
                        method: 'DELETE'
                    });
                    
                    if (!response.ok) {
                        console.error('Failed to clear categories from server');
                    }
                } catch (error) {
                    console.error('Error clearing categories from server:', error);
                }
                
                // Also clear from localStorage as backup
                localStorage.setItem('veridoc_cache_categories', JSON.stringify(window.cacheCategories));
                
                // Reload caches to reflect the changes
                await loadCaches();
                
                alert('All cache categories have been cleared');
            }
        }
        
        async function autoCategorizeAllCaches() {
            try {
                // Get current caches
                const response = await fetch('/api/caches');
                const caches = await response.json();
                
                if (!caches || caches.length === 0) {
                    alert('No caches found to categorize');
                    return;
                }
                
                // Auto-categorize each cache
                let categorizedCount = 0;
                caches.forEach(cache => {
                    const autoCategory = autoCategorizeCache(cache.original_name);
                    if (autoCategory !== 'Uncategorized') {
                        window.cacheCategories[cache.cache_id] = autoCategory;
                        categorizedCount++;
                    }
                });
                
                // Save to server
                try {
                    const response = await fetch('/api/cache-categories', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json; charset=utf-8'
                        },
                        body: JSON.stringify({
                            categories: window.cacheCategories
                        })
                    });
                    
                    if (!response.ok) {
                        console.error('Failed to save auto-categorized categories to server');
                    }
                } catch (error) {
                    console.error('Error saving auto-categorized categories to server:', error);
                }
                
                // Also save to localStorage as backup
                localStorage.setItem('veridoc_cache_categories', JSON.stringify(window.cacheCategories));
                
                // Reload the cache display
                displayCaches(caches);
                
                alert(`Successfully auto-categorized ${categorizedCount} out of ${caches.length} caches`);
                
            } catch (error) {
                console.error('Error auto-categorizing caches:', error);
                alert('Failed to auto-categorize caches: ' + error.message);
            }
        }
        
        async function deleteSelectedCaches() {
            if (selectedCaches.length === 0) {
                alert('Please select caches to delete');
                return;
            }
            
            if (!confirm('Are you sure you want to delete ' + selectedCaches.length + ' cache(s)?')) {
                return;
            }
            
            try {
                const response = await fetch('/api/caches', {
                    method: 'DELETE',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8'
                    },
                    body: JSON.stringify(selectedCaches)
                });
                
                if (!response.ok) {
                    throw new Error('Failed to delete caches');
                }
                
                alert('Caches deleted successfully');
                selectedCaches = [];
                loadCaches();
                updateStartButtonState();
            } catch (error) {
                console.error('Error deleting caches:', error);
                alert('Failed to delete caches');
            }
        }
        
        // Category Management Functions
        async function openCategoryModal() {
            const modal = document.getElementById('categoryModal');
            modal.classList.remove('hidden');
            await loadCategoryList();
        }
        
        function closeCategoryModal() {
            const modal = document.getElementById('categoryModal');
            modal.classList.add('hidden');
        }
        
        async function loadCategoryList() {
            try {
                const response = await fetch('/api/categories');
                const data = await response.json();
                const categoryList = document.getElementById('categoryList');
                
                categoryList.innerHTML = '';
                data.categories.forEach(category => {
                    const categoryItem = document.createElement('div');
                    categoryItem.className = 'category-item';
                    categoryItem.innerHTML = `
                        <div class="category-name">${category}</div>
                        <div class="category-actions-buttons">
                            <button class="btn-edit" onclick="editCategory('${category}')">Edit</button>
                            <button class="btn-delete" onclick="deleteCategory('${category}')" ${category === 'Uncategorized' ? 'disabled' : ''}>Delete</button>
                        </div>
                    `;
                    categoryList.appendChild(categoryItem);
                });
            } catch (error) {
                console.error('Error loading categories:', error);
                alert('Failed to load categories: ' + error.message);
            }
        }
        
        async function addCategory() {
            const input = document.getElementById('newCategoryName');
            const categoryName = input.value.trim();
            
            if (!categoryName) {
                alert('Please enter a category name');
                return;
            }
            
            try {
                const response = await fetch('/api/categories', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8'
                    },
                    body: JSON.stringify({ name: categoryName })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to add category');
                }
                
                input.value = '';
                await loadCategoryList();
                await loadCaches(); // Refresh cache display
                alert('Category added successfully!');
                
            } catch (error) {
                console.error('Error adding category:', error);
                alert('Failed to add category: ' + error.message);
            }
        }
        
        function editCategory(categoryName) {
            const categoryItem = event.target.closest('.category-item');
            const categoryNameDiv = categoryItem.querySelector('.category-name');
            const actionsDiv = categoryItem.querySelector('.category-actions-buttons');
            
            categoryItem.classList.add('editing');
            categoryNameDiv.innerHTML = `
                <input type="text" class="category-edit-input" value="${categoryName}">
                <div class="category-edit-buttons">
                    <button class="btn-save" onclick="saveCategoryEdit('${categoryName}')">Save</button>
                    <button class="btn-cancel" onclick="cancelCategoryEdit('${categoryName}')">Cancel</button>
                </div>
            `;
            actionsDiv.style.display = 'none';
        }
        
        async function saveCategoryEdit(oldName) {
            const categoryItem = event.target.closest('.category-item');
            const input = categoryItem.querySelector('.category-edit-input');
            const newName = input.value.trim();
            
            if (!newName) {
                alert('Please enter a category name');
                return;
            }
            
            try {
                const response = await fetch(`/api/categories/${encodeURIComponent(oldName)}`, {
                    method: 'PUT',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8'
                    },
                    body: JSON.stringify({ name: newName })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to update category');
                }
                
                await loadCategoryList();
                await loadCaches(); // Refresh cache display
                alert('Category updated successfully!');
                
            } catch (error) {
                console.error('Error updating category:', error);
                alert('Failed to update category: ' + error.message);
            }
        }
        
        function cancelCategoryEdit(categoryName) {
            const categoryItem = event.target.closest('.category-item');
            const categoryNameDiv = categoryItem.querySelector('.category-name');
            const actionsDiv = categoryItem.querySelector('.category-actions-buttons');
            
            categoryItem.classList.remove('editing');
            categoryNameDiv.textContent = categoryName;
            actionsDiv.style.display = 'flex';
        }
        
        async function deleteCategory(categoryName) {
            if (!confirm(`Are you sure you want to delete the category "${categoryName}"? All documents in this category will be moved to "Uncategorized".`)) {
                return;
            }
            
            try {
                const response = await fetch(`/api/categories/${encodeURIComponent(categoryName)}`, {
                    method: 'DELETE'
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to delete category');
                }
                
                const result = await response.json();
                await loadCategoryList();
                await loadCaches(); // Refresh cache display
                alert(result.message);
                
            } catch (error) {
                console.error('Error deleting category:', error);
                alert('Failed to delete category: ' + error.message);
            }
        }
        
        function filterCategories() {
            const searchTerm = document.getElementById('categorySearch').value.toLowerCase();
            const categoryItems = document.querySelectorAll('.category-item');
            
            categoryItems.forEach(item => {
                const categoryName = item.querySelector('.category-name').textContent.toLowerCase();
                if (categoryName.includes(searchTerm)) {
                    item.style.display = 'flex';
                } else {
                    item.style.display = 'none';
                }
            });
        }
        
        // Bulk Assignment Functions
        async function openBulkModal() {
            const modal = document.getElementById('bulkModal');
            modal.classList.remove('hidden');
            await loadBulkCategoryOptions();
            updateBulkSelectionInfo();
        }
        
        function closeBulkModal() {
            const modal = document.getElementById('bulkModal');
            modal.classList.add('hidden');
        }
        
        async function loadBulkCategoryOptions() {
            try {
                const response = await fetch('/api/categories');
                const data = await response.json();
                const select = document.getElementById('bulkCategorySelect');
                
                select.innerHTML = '<option value="">Select a category...</option>';
                data.categories.forEach(category => {
                    const option = document.createElement('option');
                    option.value = category;
                    option.textContent = category;
                    select.appendChild(option);
                });
            } catch (error) {
                console.error('Error loading categories for bulk assignment:', error);
            }
        }
        
        function updateBulkSelectionInfo() {
            const selectedCaches = getSelectedCaches();
            const infoDiv = document.getElementById('bulkSelectionInfo');
            
            if (selectedCaches.length === 0) {
                infoDiv.innerHTML = '<strong>No documents selected.</strong> Please select documents first.';
            } else {
                infoDiv.innerHTML = `<strong>${selectedCaches.length} document(s) selected</strong> for bulk category assignment.`;
            }
        }
        
        function getSelectedCaches() {
            const checkboxes = document.querySelectorAll('.cache-checkbox:checked');
            return Array.from(checkboxes).map(cb => cb.value);
        }
        
        async function performBulkAssignment() {
            const categorySelect = document.getElementById('bulkCategorySelect');
            const selectedCategory = categorySelect.value;
            const selectedCaches = getSelectedCaches();
            
            if (!selectedCategory) {
                alert('Please select a category');
                return;
            }
            
            if (selectedCaches.length === 0) {
                alert('Please select documents to assign');
                return;
            }
            
            try {
                const response = await fetch('/api/bulk-assign-categories', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8'
                    },
                    body: JSON.stringify({
                        cache_ids: selectedCaches,
                        category: selectedCategory
                    })
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Failed to assign categories');
                }
                
                const result = await response.json();
                await loadCaches(); // Refresh cache display
                closeBulkModal();
                alert(result.message);
                
            } catch (error) {
                console.error('Error performing bulk assignment:', error);
                alert('Failed to assign categories: ' + error.message);
            }
        }
        
        // Update the existing loadCaches function to use server-side categories
        async function loadCaches() {
            try {
                // Load categories from server
                try {
                    const categoriesResponse = await fetch('/api/cache-categories');
                    if (categoriesResponse.ok) {
                        const categoriesData = await categoriesResponse.json();
                        window.cacheCategories = categoriesData.categories || {};
                        window.allCategories = categoriesData.all_categories || [];
                        console.log('Loaded categories from server:', window.cacheCategories);
                    }
                } catch (error) {
                    console.warn('Failed to load categories from server, using localStorage:', error);
                    // Fallback to localStorage
                    const savedCategories = localStorage.getItem('veridoc_cache_categories');
                    if (savedCategories) {
                        window.cacheCategories = JSON.parse(savedCategories);
                    } else {
                        window.cacheCategories = {};
                    }
                }
                
                const response = await fetch('/api/caches');
                const caches = await response.json();
                displayCaches(caches);
            } catch (error) {
                console.error('Error loading caches:', error);
                alert('Failed to load caches: ' + error.message);
            }
        }
        
        // Update the generateCategoryOptions function to use server-side categories
        function generateCategoryOptions(cacheId) {
            let options = '<option value="Uncategorized">Uncategorized</option>';
            
            if (window.allCategories) {
                window.allCategories.forEach(category => {
                    if (category !== 'Uncategorized') {
                        const selected = window.cacheCategories[cacheId] === category ? 'selected' : '';
                        options += `<option value="${category}" ${selected}>${category}</option>`;
                    }
                });
            }
            
            return options;
        }
        
        // Close modals when clicking outside
        window.onclick = function(event) {
            const categoryModal = document.getElementById('categoryModal');
            const bulkModal = document.getElementById('bulkModal');
            
            if (event.target === categoryModal) {
                closeCategoryModal();
            }
            if (event.target === bulkModal) {
                closeBulkModal();
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.get("/mobile", response_class=HTMLResponse)
async def get_mobile_page():
    """Serve the mobile HTML page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeriDoc AI - Mobile</title>
    <link rel="stylesheet" href="/static/styles.css">
    <style>
        /* Mobile-specific styles */
        .mobile-container {
            max-width: 100%;
            padding: 10px;
            margin: 0;
        }
        
        .mobile-header {
            text-align: center;
            margin-bottom: 20px;
        }
        
        .mobile-header h1 {
            font-size: 24px;
            margin: 0 0 5px 0;
        }
        
        .mobile-header p {
            font-size: 14px;
            margin: 0;
            color: #666;
        }
        
        .mobile-section {
            margin-bottom: 30px;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        
        .mobile-section h3 {
            margin: 0 0 15px 0;
            font-size: 18px;
            color: #333;
        }
        
        .mobile-section p {
            margin: 0 0 15px 0;
            font-size: 14px;
            color: #666;
        }
        
        .mobile-input {
            width: 100%;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 6px;
            font-size: 16px;
            margin-bottom: 15px;
            box-sizing: border-box;
        }
        
        .mobile-btn {
            width: 100%;
            padding: 12px;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            margin-bottom: 10px;
            transition: background-color 0.2s;
        }
        
        .mobile-btn.primary {
            background-color: #007bff;
            color: white;
        }
        
        .mobile-btn.primary:hover {
            background-color: #0056b3;
        }
        
        .mobile-btn.secondary {
            background-color: #6c757d;
            color: white;
        }
        
        .mobile-btn.secondary:hover {
            background-color: #545b62;
        }
        
        .mobile-btn:disabled {
            background-color: #ccc;
            cursor: not-allowed;
        }
        
        .mobile-conversation {
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 15px;
            margin-top: 15px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        .mobile-message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 6px;
        }
        
        .mobile-message.user {
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
        }
        
        .mobile-message.assistant {
            background-color: #f3e5f5;
            border-left: 4px solid #9c27b0;
        }
        
        .mobile-message .timestamp {
            font-size: 12px;
            color: #666;
            margin-bottom: 5px;
        }
        
        .mobile-message .content {
            font-size: 14px;
            line-height: 1.4;
        }
        
        .mobile-knowledge-list {
            background: white;
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 10px;
            max-height: 200px;
            overflow-y: auto;
        }
        
        .mobile-knowledge-item {
            padding: 8px;
            border-bottom: 1px solid #eee;
            font-size: 14px;
        }
        
        .mobile-knowledge-item:last-child {
            border-bottom: none;
        }
        
        .mobile-file-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 8px;
        }
        
        .mobile-file-name {
            flex: 1;
            font-weight: 500;
        }
        
        .mobile-category-badge {
            background: #e3f2fd;
            color: #1976d2;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 11px;
            font-weight: 500;
            white-space: nowrap;
            max-width: 120px;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        
        .mobile-empty-state {
            text-align: center;
            padding: 20px;
            color: #666;
            font-style: italic;
        }
        
        .mobile-category-group {
            margin-bottom: 20px;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            overflow: hidden;
        }
        
        .mobile-category-header {
            background: #f5f5f5;
            padding: 12px 15px;
            font-weight: 600;
            font-size: 14px;
            color: #333;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .mobile-category-files {
            background: white;
        }
        
        .mobile-category-files .mobile-knowledge-item {
            border-bottom: 1px solid #f0f0f0;
            margin: 0;
        }
        
        .mobile-category-files .mobile-knowledge-item:last-child {
            border-bottom: none;
        }
        
        .mobile-category-header {
            cursor: pointer;
            -webkit-user-select: none;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .mobile-category-header:hover {
            background: #e8e8e8;
        }
        
        .mobile-category-toggle {
            font-size: 12px;
            color: #666;
            transition: transform 0.2s ease;
        }
        
        .mobile-category-header.collapsed .mobile-category-toggle {
            transform: rotate(-90deg);
        }
        
        .mobile-confidence {
            margin-top: 8px;
            font-size: 12px;
            color: #666;
        }
        
        .mobile-key-points {
            margin-top: 8px;
            font-size: 12px;
        }
        
        .mobile-key-points ul {
            margin: 4px 0;
            padding-left: 20px;
        }
        
        .mobile-key-points li {
            margin-bottom: 2px;
            color: #666;
        }
        
        .mobile-conversation-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 10px;
            padding-bottom: 10px;
            border-bottom: 1px solid #e0e0e0;
        }
        
        .mobile-conversation-header h4 {
            margin: 0;
            flex: 1;
            font-size: 16px;
            color: #333;
        }
        
        .mobile-conversation-copy-btn {
            background: #f3f4f6;
            border: 1px solid #d1d5db;
            border-radius: 4px;
            padding: 6px 8px;
            cursor: pointer;
            font-size: 14px;
            color: #6b7280;
            transition: all 0.2s ease;
            opacity: 0.8;
            margin-left: 10px;
        }
        
        .mobile-conversation-copy-btn:hover {
            background: #e5e7eb;
            opacity: 1;
        }
        
        .mobile-conversation-copy-btn.copied {
            background: #10b981;
            color: white;
            border-color: #10b981;
        }
    </style>
</head>
<body>
    <div class="mobile-container">
        <div class="mobile-header">
            <h1>VeriDoc AI</h1>
            <p>Mobile Interface</p>
        </div>
        
        <!-- Ask Me Section -->
        <div class="mobile-section">
            <h3>🤔 Ask Me</h3>
            <p>Ask questions about your knowledge database and get AI-powered answers.</p>
            
            <textarea id="mobileAskMeInput" class="mobile-input" placeholder="Type your question here..." rows="4"></textarea>
            
            <button class="mobile-btn primary" id="mobileAskMeBtn" disabled>💬 Ask Question</button>
            <button class="mobile-btn secondary" id="mobileNewConversationBtn">🆕 New Conversation</button>
            
            <div id="mobileConversationHistory" class="mobile-conversation mobile-conversation-hidden">
                <div class="mobile-conversation-header">
                    <h4>💬 Conversation History</h4>
                    <button class="mobile-conversation-copy-btn" onclick="copyMobileConversation()" title="Copy entire conversation">📋</button>
                </div>
                <div id="mobileConversationMessages"></div>
            </div>
        </div>
        
        <!-- Knowledge Database Section -->
        <div class="mobile-section">
            <h3>📚 Knowledge Database</h3>
            <p>View your indexed knowledge files.</p>
            
            <div id="mobileKnowledgeList" class="mobile-knowledge-list">
                <div class="mobile-empty-state">Loading knowledge database...</div>
            </div>
        </div>
    </div>

    <script>
        // Mobile-specific JavaScript
        let mobileConversationHistory = [];
        let mobileConversationId = '';
        
        // Initialize mobile interface
        document.addEventListener('DOMContentLoaded', function() {
            loadMobileKnowledgeDatabase();
            setupMobileEventListeners();
        });
        
        function setupMobileEventListeners() {
            const askMeInput = document.getElementById('mobileAskMeInput');
            const askMeBtn = document.getElementById('mobileAskMeBtn');
            const newConversationBtn = document.getElementById('mobileNewConversationBtn');
            
            // Enable/disable ask button based on input
            askMeInput.addEventListener('input', function() {
                askMeBtn.disabled = !this.value.trim();
            });
            
            // Ask question
            askMeBtn.addEventListener('click', function() {
                const question = askMeInput.value.trim();
                if (question) {
                    askMobileQuestion(question);
                }
            });
            
            // New conversation
            newConversationBtn.addEventListener('click', function() {
                mobileConversationHistory = [];
                mobileConversationId = '';
                document.getElementById('mobileConversationMessages').innerHTML = '';
                document.getElementById('mobileConversationHistory').classList.add('mobile-conversation-hidden');
                askMeInput.value = '';
                askMeBtn.disabled = true;
            });
            
            // Enter key to submit
            askMeInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    if (!askMeBtn.disabled) {
                        askMeBtn.click();
                    }
                }
            });
        }
        
        async function loadMobileKnowledgeDatabase() {
            try {
                // Fetch both caches and categories
                const [cachesResponse, categoriesResponse] = await Promise.all([
                    fetch('/api/caches'),
                    fetch('/api/cache-categories')
                ]);
                
                const cachesData = await cachesResponse.json();
                const categoriesData = await categoriesResponse.json();
                

                
                const knowledgeList = document.getElementById('mobileKnowledgeList');
                
                if (cachesData && cachesData.length > 0) {
                    // Group files by category
                    const groupedFiles = {};
                    
                    cachesData.forEach(cache => {
                        const category = categoriesData.categories[cache.cache_id] || 'Uncategorized';
                        if (!groupedFiles[category]) {
                            groupedFiles[category] = [];
                        }
                        groupedFiles[category].push(cache);
                    });
                    
                    // Generate HTML with category groups
                    const categoryGroups = Object.keys(groupedFiles).sort();
                    const htmlContent = categoryGroups.map(category => {
                        const files = groupedFiles[category];
                        const filesHtml = files.map(cache => 
                            `<div class="mobile-knowledge-item">
                                <div class="mobile-file-info">
                                    <span class="mobile-file-name">📄 ${cache.original_name}</span>
                                </div>
                            </div>`
                        ).join('');
                        
                        return `
                            <div class="mobile-category-group">
                                <div class="mobile-category-header collapsed" onclick="toggleMobileCategory(this)">
                                    <span class="mobile-category-title">${category} (${files.length})</span>
                                    <span class="mobile-category-toggle">▶</span>
                                </div>
                                <div class="mobile-category-files hidden">
                                    ${filesHtml}
                                </div>
                            </div>
                        `;
                    }).join('');
                    
                    knowledgeList.innerHTML = htmlContent;
                } else {
                    knowledgeList.innerHTML = '<div class="mobile-empty-state">No knowledge files found</div>';
                }
            } catch (error) {
                console.error('Error loading knowledge database:', error);
                document.getElementById('mobileKnowledgeList').innerHTML = 
                    '<div class="mobile-empty-state">Error loading knowledge database</div>';
            }
        }
        
        function toggleMobileCategory(headerElement) {
            const categoryFiles = headerElement.nextElementSibling;
            const toggleIcon = headerElement.querySelector('.mobile-category-toggle');
            
            if (categoryFiles.classList.contains('hidden')) {
                categoryFiles.classList.remove('hidden');
                toggleIcon.textContent = '▼';
                headerElement.classList.remove('collapsed');
            } else {
                categoryFiles.classList.add('hidden');
                toggleIcon.textContent = '▶';
                headerElement.classList.add('collapsed');
            }
        }
        
        async function askMobileQuestion(question) {
            const askMeBtn = document.getElementById('mobileAskMeBtn');
            const askMeInput = document.getElementById('mobileAskMeInput');
            
            // Disable button and show loading
            askMeBtn.disabled = true;
            askMeBtn.textContent = '⏳ Processing...';
            
            // Add user message to conversation
            const userMessage = {
                role: 'user',
                content: question,
                timestamp: new Date().toLocaleTimeString()
            };
            mobileConversationHistory.push(userMessage);
            
            // Display conversation
            displayMobileConversation();
            
            try {
                // Get all available caches for mobile (no selection needed)
                const cachesResponse = await fetch('/api/caches');
                const cachesData = await cachesResponse.json();
                const selectedCaches = cachesData ? cachesData.map(cache => cache.cache_id) : [];
                
                const response = await fetch('/api/ask-me', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8',
                    },
                    body: JSON.stringify({
                        question: question,
                        selected_caches: selectedCaches,
                        conversation_history: mobileConversationHistory.slice(0, -1), // Exclude current message
                        conversation_id: mobileConversationId || ''
                    })
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    // Add assistant response to conversation
                    const assistantMessage = {
                        role: 'assistant',
                        content: data.answer || 'No answer provided',
                        timestamp: new Date().toLocaleTimeString(),
                        data: data
                    };
                    mobileConversationHistory.push(assistantMessage);
                } else {
                    // Add error message
                    const errorMessage = {
                        role: 'assistant',
                        content: 'Sorry, I encountered an error while processing your question. Please try again.',
                        timestamp: new Date().toLocaleTimeString()
                    };
                    mobileConversationHistory.push(errorMessage);
                }
                
            } catch (error) {
                console.error('Error asking question:', error);
                const errorMessage = {
                    role: 'assistant',
                    content: 'Sorry, I encountered an error while processing your question. Please try again.',
                    timestamp: new Date().toLocaleTimeString()
                };
                mobileConversationHistory.push(errorMessage);
            }
            
            // Update display and reset input
            displayMobileConversation();
            askMeInput.value = '';
            askMeBtn.disabled = true;
            askMeBtn.textContent = '💬 Ask Question';
        }
        
        function displayMobileConversation() {
            const conversationContainer = document.getElementById('mobileConversationHistory');
            const messagesContainer = document.getElementById('mobileConversationMessages');
            
            if (mobileConversationHistory.length > 0) {
                conversationContainer.classList.remove('mobile-conversation-hidden');
                
                messagesContainer.innerHTML = mobileConversationHistory.map(message => {
                    let messageHtml = `
                        <div class="mobile-message ${message.role}">
                            <div class="timestamp">${message.timestamp}</div>
                            <div class="content">${message.content}</div>`;
                    
                    // Add confidence and key points for assistant messages with data
                    if (message.role === 'assistant' && message.data) {
                        if (message.data.confidence) {
                            messageHtml += `<div class="mobile-confidence"><strong>Confidence:</strong> ${message.data.confidence}</div>`;
                        }
                        if (message.data.key_points && message.data.key_points.length > 0) {
                            messageHtml += `<div class="mobile-key-points"><strong>Key Points:</strong><ul>`;
                            message.data.key_points.forEach(point => {
                                messageHtml += `<li>${point}</li>`;
                            });
                            messageHtml += `</ul></div>`;
                        }
                    }
                    
                    messageHtml += `</div>`;
                    return messageHtml;
                }).join('');
                
                // Scroll to bottom
                messagesContainer.scrollTop = messagesContainer.scrollHeight;
            } else {
                conversationContainer.classList.add('mobile-conversation-hidden');
            }
        }
        
        function copyMobileConversation() {
            console.log('copyMobileConversation function called');
            
            // Check if clipboard API is available
            if (!navigator.clipboard) {
                console.log('Clipboard API not available, using fallback method');
            }
            
            const messagesContainer = document.getElementById('mobileConversationMessages');
            console.log('messagesContainer:', messagesContainer);
            
            if (!messagesContainer) {
                console.error('mobileConversationMessages container not found');
                alert('Error: Conversation container not found');
                return;
            }
            
            const messages = messagesContainer.querySelectorAll('.mobile-message');
            console.log('Found messages:', messages.length);
            
            if (messages.length === 0) {
                console.log('No messages to copy');
                alert('No conversation to copy');
                return;
            }
            
            let conversationText = '';
            
            messages.forEach((message, index) => {
                const role = message.classList.contains('user') ? 'User' : 'Assistant';
                const time = message.querySelector('.timestamp')?.textContent || '';
                const content = message.querySelector('.content')?.textContent || '';
                
                console.log(`Message ${index + 1}:`, { role, time, content: content.substring(0, 50) + '...' });
                
                conversationText += `[${time}] ${role}:\n${content}\n\n`;
            });
            
            // Remove trailing newlines
            conversationText = conversationText.trim();
            console.log('Final conversation text length:', conversationText.length);
            console.log('Conversation text preview:', conversationText.substring(0, 200) + '...');
            
            const copyBtn = document.querySelector('.mobile-conversation-copy-btn');
            console.log('Copy button found:', copyBtn);
            
            if (!copyBtn) {
                console.error('Copy button not found');
                alert('Error: Copy button not found');
                return;
            }
            
            // Try modern clipboard API first
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(conversationText).then(() => {
                    console.log('Successfully copied to clipboard using modern API');
                    // Visual feedback
                    copyBtn.classList.add('copied');
                    copyBtn.textContent = '✓';
                    
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.textContent = '📋';
                    }, 2000);
                }).catch(err => {
                    console.error('Failed to copy conversation with modern API: ', err);
                    console.log('Trying fallback method...');
                    copyWithFallback(conversationText, copyBtn);
                });
            } else {
                console.log('Modern clipboard API not available, using fallback');
                copyWithFallback(conversationText, copyBtn);
            }
        }
        
        function copyWithFallback(text, copyBtn) {
            try {
                const textArea = document.createElement('textarea');
                textArea.value = text;
                textArea.style.position = 'fixed';
                textArea.style.left = '-999999px';
                textArea.style.top = '-999999px';
                document.body.appendChild(textArea);
                textArea.focus();
                textArea.select();
                
                const successful = document.execCommand('copy');
                document.body.removeChild(textArea);
                
                if (successful) {
                    console.log('Successfully copied using fallback method');
                    // Visual feedback
                    copyBtn.classList.add('copied');
                    copyBtn.textContent = '✓';
                    
                    setTimeout(() => {
                        copyBtn.classList.remove('copied');
                        copyBtn.textContent = '📋';
                    }, 2000);
                } else {
                    console.error('Fallback copy method failed');
                    alert('Failed to copy conversation. Please try selecting and copying manually.');
                }
            } catch (err) {
                console.error('Error in fallback copy method:', err);
                alert('Failed to copy conversation. Please try selecting and copying manually.');
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file with proper validation and error handling"""
    try:
        # Validate file
        if not file.filename:
            logger.warning("Upload attempted with no filename")
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Check file extension
        allowed_extensions = {'.pdf', '.docx', '.xlsx', '.txt', '.xls', '.doc'}
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in allowed_extensions:
            logger.warning(f"Invalid file extension: {file_ext}")
            raise HTTPException(
                status_code=400, 
                detail=f"File type {file_ext} not supported. Allowed: {', '.join(allowed_extensions)}"
            )
        
        # Create sanitized filename
        safe_filename = "".join(c for c in file.filename if c.isalnum() or c in '._-')
        file_path = UPLOAD_DIR / safe_filename
        
        # Check file size (limit to 50MB)
        max_size = 50 * 1024 * 1024  # 50MB
        file_size = 0
        
        logger.info(f"Uploading file: {safe_filename}")
        
        with open(file_path, "wb") as buffer:
            while chunk := await file.read(8192):  # Read in 8KB chunks
                file_size += len(chunk)
                if file_size > max_size:
                    # Clean up partial file
                    buffer.close()
                    if file_path.exists():
                        file_path.unlink()
                    logger.warning(f"File too large: {file_size} bytes")
                    raise HTTPException(status_code=413, detail="File too large (max 50MB)")
                buffer.write(chunk)
        
        logger.info(f"Successfully uploaded {safe_filename} ({file_size} bytes)")
        return {
            "message": "File uploaded successfully", 
            "filename": safe_filename,
            "size": file_size
        }
        
    except HTTPException:
        raise
    except OSError as e:
        logger.error(f"File system error during upload: {e}")
        raise HTTPException(status_code=500, detail="File system error")
    except Exception as e:
        logger.error(f"Unexpected error during upload: {e}")
        raise HTTPException(status_code=500, detail="Upload failed")

@app.post("/api/process")
async def start_processing(background_tasks: BackgroundTasks, request: Request):
    """Start the verification process with proper validation"""
    try:
        # Parse and validate request data
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"Invalid JSON in request: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON data")
        
        selected_caches = data.get("selected_caches", [])
        statements_file = data.get("statements_file", "")
        knowledge_files = data.get("knowledge_files", [])
        selected_statements = data.get("selected_statements", [])
        
        # Validate inputs
        if not selected_caches and not knowledge_files:
            logger.warning("No knowledge sources selected")
            raise HTTPException(status_code=400, detail="No knowledge sources selected")
        
        if not statements_file and not selected_statements:
            logger.warning("No statements file or statements selected")
            raise HTTPException(status_code=400, detail="No statements to process")
        
        # Check if statements file exists
        if statements_file:
            statements_path = UPLOAD_DIR / statements_file
            if not statements_path.exists():
                logger.error(f"Statements file not found: {statements_file}")
                raise HTTPException(status_code=404, detail=f"Statements file '{statements_file}' not found")
        
        logger.info(f"Starting verification process with {len(selected_caches)} caches and {len(knowledge_files)} files")
        
        # Reset processing status
        global processing_status
        processing_status = {
            "status": "idle",
            "progress": 0,
            "current_step": "",
            "processed_items": 0,
            "total_items": 0,
            "message": "",
            "logs": []
        }
        
        # Start background processing
        background_tasks.add_task(
            process_verification_background, 
            selected_caches, 
            statements_file,
            selected_statements
        )
        
        return {"message": "Processing started successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

@app.get("/api/caches")
async def get_caches():
    """Get list of available caches"""
    try:
        caches = list_available_caches()
        from fastapi.responses import JSONResponse
        return JSONResponse(content=caches, headers={"Content-Type": "application/json; charset=utf-8"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch caches: {str(e)}")



@app.get("/api/cache-categories")
async def get_cache_categories():
    """Get all cache categories"""
    try:
        return {"categories": cache_categories, "all_categories": all_categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch cache categories: {str(e)}")

@app.post("/api/cache-categories")
async def update_cache_categories(request: Request):
    """Update cache categories"""
    try:
        data = await request.json()
        global cache_categories
        
        # Update categories
        cache_categories.update(data.get("categories", {}))
        
        # Save to file
        save_cache_categories()
        
        logger.info(f"Updated cache categories: {len(cache_categories)} total categories")
        return {"message": "Cache categories updated successfully", "categories": cache_categories}
        
    except Exception as e:
        logger.error(f"Error updating cache categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update cache categories: {str(e)}")

@app.delete("/api/cache-categories")
async def clear_cache_categories():
    """Clear all cache categories"""
    try:
        global cache_categories
        cache_categories = {}
        save_cache_categories()
        
        logger.info("Cleared all cache categories")
        return {"message": "All cache categories cleared successfully"}
        
    except Exception as e:
        logger.error(f"Error clearing cache categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache categories: {str(e)}")

@app.get("/api/categories")
async def get_categories():
    """Get all available categories"""
    try:
        return {"categories": all_categories}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch categories: {str(e)}")

@app.post("/api/categories")
async def add_category(request: Request):
    """Add a new category"""
    global all_categories
    try:
        data = await request.json()
        category_name = data.get("name", "").strip()
        
        if not category_name:
            raise HTTPException(status_code=400, detail="Category name is required")
        
        if category_name in all_categories:
            raise HTTPException(status_code=400, detail="Category already exists")
        
        all_categories.append(category_name)
        save_category_list()
        
        logger.info(f"Added new category: {category_name}")
        return {"message": "Category added successfully", "categories": all_categories}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to add category: {str(e)}")

@app.put("/api/categories/{old_name}")
async def update_category(old_name: str, request: Request):
    """Update a category name"""
    global all_categories, cache_categories
    try:
        data = await request.json()
        new_name = data.get("name", "").strip()
        
        if not new_name:
            raise HTTPException(status_code=400, detail="Category name is required")
        
        if new_name in all_categories and new_name != old_name:
            raise HTTPException(status_code=400, detail="Category name already exists")
        
        # Update category list
        if old_name in all_categories:
            index = all_categories.index(old_name)
            all_categories[index] = new_name
            save_category_list()
        
        # Update all cache assignments
        updated_categories = {}
        for cache_id, category in cache_categories.items():
            if category == old_name:
                updated_categories[cache_id] = new_name
            else:
                updated_categories[cache_id] = category
        
        cache_categories = updated_categories
        save_cache_categories()
        
        logger.info(f"Updated category: {old_name} -> {new_name}")
        return {"message": "Category updated successfully", "categories": all_categories}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update category: {str(e)}")

@app.delete("/api/categories/{category_name}")
async def delete_category(category_name: str):
    """Delete a category and reassign its documents to Uncategorized"""
    try:
        global all_categories, cache_categories
        
        if category_name not in all_categories:
            raise HTTPException(status_code=404, detail="Category not found")
        
        if category_name == "Uncategorized":
            raise HTTPException(status_code=400, detail="Cannot delete Uncategorized category")
        
        # Remove from category list
        all_categories.remove(category_name)
        save_category_list()
        
        # Reassign all documents from this category to Uncategorized
        updated_categories = {}
        reassigned_count = 0
        for cache_id, category in cache_categories.items():
            if category == category_name:
                updated_categories[cache_id] = "Uncategorized"
                reassigned_count += 1
            else:
                updated_categories[cache_id] = category
        
        cache_categories = updated_categories
        save_cache_categories()
        
        logger.info(f"Deleted category: {category_name}, reassigned {reassigned_count} documents to Uncategorized")
        return {
            "message": f"Category deleted successfully. {reassigned_count} documents reassigned to Uncategorized.",
            "categories": all_categories
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting category: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete category: {str(e)}")

@app.post("/api/bulk-assign-categories")
async def bulk_assign_categories(request: Request):
    """Bulk assign categories to multiple caches"""
    try:
        data = await request.json()
        cache_ids = data.get("cache_ids", [])
        category_name = data.get("category", "").strip()
        
        if not cache_ids:
            raise HTTPException(status_code=400, detail="No cache IDs provided")
        
        if not category_name:
            raise HTTPException(status_code=400, detail="Category name is required")
        
        if category_name not in all_categories:
            raise HTTPException(status_code=400, detail="Category does not exist")
        
        global cache_categories
        assigned_count = 0
        
        for cache_id in cache_ids:
            cache_categories[cache_id] = category_name
            assigned_count += 1
        
        save_cache_categories()
        
        logger.info(f"Bulk assigned {assigned_count} caches to category: {category_name}")
        return {
            "message": f"Successfully assigned {assigned_count} caches to {category_name}",
            "assigned_count": assigned_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error bulk assigning categories: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to bulk assign categories: {str(e)}")

@app.delete("/api/caches")
async def delete_caches(request: Request):
    """Delete selected caches"""
    try:
        cache_ids = await request.json()
        success_count = 0
        for cache_id in cache_ids:
            if delete_cache(cache_id):
                success_count += 1
        
        return {"message": f"Deleted {success_count}/{len(cache_ids)} caches"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete caches: {str(e)}")

@app.get("/api/results")
async def get_results():
    """Get verification results"""
    try:
        results_file = "verification_results.xlsx"
        if not os.path.exists(results_file):
            return []
        
        # Read Excel file and return as JSON
        df = pd.read_excel(results_file)
        
        # Handle NaN values more robustly
        def clean_value(val):
            if pd.isna(val) or val == 'nan' or val == 'NaN':
                return None
            if isinstance(val, float) and (val != val or val == float('inf') or val == float('-inf')):
                return None
            return val
        
        # Clean all values in the dataframe
        cleaned_data = []
        for _, row in df.iterrows():
            cleaned_row = {}
            for col in df.columns:
                cleaned_row[col] = clean_value(row[col])
            cleaned_data.append(cleaned_row)
        
        return cleaned_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")

@app.get("/api/results/download")
async def download_results():
    """Download verification results as Excel file"""
    try:
        results_file = "verification_results.xlsx"
        if not os.path.exists(results_file):
            raise HTTPException(status_code=404, detail="Results file not found")
        
        return FileResponse(
            results_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename="verification_results.xlsx"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to download results: {str(e)}")

@app.post("/api/statements-data")
async def get_statements_data(request: Request):
     """Get statements data from uploaded file"""
     try:
         data = await request.json()
         filename = data.get("filename", "")
         
         if not filename:
             raise HTTPException(status_code=400, detail="Filename is required")
         
         file_path = UPLOAD_DIR / filename
         if not file_path.exists():
             raise HTTPException(status_code=404, detail="File not found")
         
         # Read statements using the existing function
         statements = read_statements(str(file_path))
         
         return {"statements": statements}
     except Exception as e:
         raise HTTPException(status_code=500, detail=f"Failed to load statements data: {str(e)}")

@app.post("/api/index-files")
async def index_files(request: Request):
    """Index and cache uploaded knowledge files"""
    try:
        data = await request.json()
        knowledge_files = data.get("knowledge_files", [])
        
        logger.info(f"Indexing request received for {len(knowledge_files)} files: {knowledge_files}")
        
        if not knowledge_files:
            raise HTTPException(status_code=400, detail="No files to index")
        
        # Load API credentials
        try:
            api_key, base_url = load_api_credentials("api.txt")
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url)
            logger.info("API credentials loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load API credentials: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to load API credentials: {str(e)}")
        
        indexed_count = 0
        failed_files = []
        
        for filename in knowledge_files:
            try:
                file_path = UPLOAD_DIR / filename
                logger.info(f"Processing file: {filename} at path: {file_path}")
                
                if not file_path.exists():
                    logger.warning(f"File not found: {file_path}")
                    failed_files.append(f"{filename} (file not found)")
                    continue
                
                logger.info(f"Building index for: {filename}")
                
                # Build or load index for this file
                embeddings, chunks, index_id = build_or_load_index(
                    corpus_path=str(file_path),
                    embedding_model="text-embedding-3-large",
                    client=client,
                    chunk_size_chars=4000,
                    overlap_chars=500,
                )
                
                indexed_count += 1
                logger.info(f"Successfully indexed: {filename} (ID: {index_id}, chunks: {len(chunks)})")
                
            except Exception as e:
                error_msg = f"Failed to index {filename}: {str(e)}"
                logger.error(error_msg)
                failed_files.append(f"{filename} ({str(e)})")
                continue
        
        result_msg = f"Indexing completed. Successfully indexed {indexed_count} files."
        if failed_files:
            result_msg += f" Failed files: {', '.join(failed_files)}"
        
        logger.info(result_msg)
        return {"message": result_msg, "indexed_count": indexed_count, "failed_files": failed_files}
        
    except Exception as e:
        error_msg = f"Failed to index files: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail=error_msg)

@app.post("/api/ask-me")
async def ask_me(request: Request):
    """Answer questions based on the knowledge database and cached context files"""
    try:
        logger.info("ASK ME API CALLED")
        data = await request.json()
        logger.debug(f"Received data: {data}")
        
        question = data.get("question", "").strip()
        selected_caches = data.get("selected_caches", [])
        conversation_id = data.get("conversation_id", "")
        
        logger.debug(f"Question: '{question}'")
        logger.debug(f"Selected caches: {selected_caches}")
        logger.debug(f"Conversation ID: {conversation_id}")
        
        if not question:
            logger.warning("No question provided")
            raise HTTPException(status_code=400, detail="Question is required")
        
        if not selected_caches and not conversation_id:
            logger.warning("No caches or conversation context selected")
            raise HTTPException(status_code=400, detail="No knowledge database or context selected")
        
        # Load API credentials
        api_key, base_url = load_api_credentials("api.txt")
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Initialize context sources
        all_embeddings = []
        all_chunks = []
        context_sources = []
        
        # Load knowledge database caches
        if selected_caches:
            embeddings, chunks, cache_names = load_multiple_caches(selected_caches)
            all_embeddings.extend(embeddings)
            all_chunks.extend(chunks)
            context_sources.extend(cache_names)
            logger.info(f"Loaded {len(embeddings)} embeddings and {len(chunks)} chunks from {len(selected_caches)} knowledge caches")
        
        # Load Ask Me context cache
        if conversation_id:
            context_cache = get_ask_me_context_cache(conversation_id)
            if context_cache:
                all_embeddings.extend(context_cache["embeddings"])
                all_chunks.extend(context_cache["chunks"])
                context_sources.append(f"Ask Me Context ({len(context_cache['chunks'])} chunks)")
                logger.info(f"Loaded {len(context_cache['embeddings'])} embeddings and {len(context_cache['chunks'])} chunks from Ask Me context cache")
            else:
                logger.warning(f"No context cache found for conversation {conversation_id}")
        
        if not all_embeddings:
            logger.warning("No embeddings available")
            return {"answer": "I couldn't find relevant information to answer your question."}
        
        # Find relevant chunks for the question
        question_embedding = embed_single_text(client, "text-embedding-3-large", question)
        
        # Calculate similarities
        similarities = []
        for i, embedding in enumerate(all_embeddings):
            similarity = cosine_similarity([question_embedding], [embedding])[0][0]
            similarities.append((similarity, i))
        
        # Get top relevant chunks
        similarities.sort(reverse=True)
        logger.debug(f"Top 5 similarities: {similarities[:5]}")
        
        top_chunks = []
        for similarity, idx in similarities[:10]:  # Top 10 most relevant
            if similarity > 0.1:  # Minimum relevance threshold
                chunk_text = all_chunks[idx] if isinstance(all_chunks[idx], str) else all_chunks[idx][1]
                top_chunks.append(chunk_text)
        
        logger.info(f"Found {len(top_chunks)} relevant chunks from {len(context_sources)} sources")
        
        if not top_chunks:
            logger.warning("No relevant chunks found, returning default message")
            return {"answer": "I couldn't find relevant information in the knowledge database to answer your question."}
        
        # Create context from relevant chunks - use all top chunks
        context_chunks = top_chunks[:10]  # Use top 10 chunks
        context = "\n\n".join(context_chunks)
        
        # Limit context to ~40000 characters to stay well within token limits
        if len(context) > 40000:
            context = context[:40000] + "..."
        
        logger.debug(f"Context length: {len(context)} characters")
        
        # Create prompt for the model
        prompt = f"""You are experienced UK solicitor asked to analyse question on a set of available knowledge base and answer the question in detailed explanatory manner.

{context}

Question: {question}

Please provide your answer in the following JSON format:
{{
    "answer": "your detailed answer here",
    "confidence": "high/medium/low",
    "key_points": ["point1", "point2", "point3"]
}}"""
        
        logger.debug(f"Prompt length: {len(prompt)} characters")
        
        # Get answer from the model
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {"role": "system", "content": "You are experienced UK solicitor asked to analyse question on a set of available knowledge base and answer the question in detailed explanatory manner."},
                    {"role": "user", "content": prompt}
                ],
                max_completion_tokens=4000,  # Limit to ~250 words
                reasoning_effort="medium",  # Use medium reasoning effort
                verbosity="low"  # Use medium verbosity
            )
            
            if response.choices and len(response.choices) > 0:
                answer = response.choices[0].message.content.strip()
                logger.debug(f"Model response content: '{answer}'")
                
                # Try to parse JSON response
                try:
                    import json
                    parsed_response = json.loads(answer)
                    return parsed_response
                except json.JSONDecodeError:
                    # If not valid JSON, return as plain answer
                    return {"answer": answer, "confidence": "medium", "key_points": []}
            else:
                answer = "No response generated from the model."
                return {"answer": answer, "confidence": "low", "key_points": []}
            
        except Exception as model_error:
            logger.error(f"Model API error: {model_error}")
            return {"answer": f"Error generating response: {str(model_error)}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to answer question: {str(e)}")

@app.post("/api/ask-me-upload")
async def ask_me_upload(request: Request):
    """Upload additional context files for Ask Me module with caching, indexing, and embedding"""
    try:
        form = await request.form()
        files = form.getlist("files")
        conversation_id = form.get("conversation_id", "")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Generate conversation ID if not provided
        if not conversation_id:
            conversation_id = generate_conversation_id()
        
        uploaded_files = []
        
        for file in files:
            if not file.filename:
                continue
                
            # Save file temporarily
            file_path = UPLOAD_DIR / file.filename
            with open(file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            
            # Extract text from file
            try:
                extracted_text = extract_text_from_file(str(file_path))
                uploaded_files.append({
                    "filename": file.filename,
                    "content": extracted_text
                })
                logger.info(f"Successfully processed {file.filename}")
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {e}")
                # Still add the file but with error message
                uploaded_files.append({
                    "filename": file.filename,
                    "content": f"Error extracting text from file: {str(e)}"
                })
        
        # Create cache for the uploaded files
        if uploaded_files:
            try:
                cache_data = create_ask_me_context_cache(conversation_id, uploaded_files)
                logger.info(f"Created context cache for conversation {conversation_id}")
            except Exception as cache_error:
                logger.error(f"Failed to create context cache: {cache_error}")
                # Continue without caching, but log the error
        
        return {
            "uploaded_files": uploaded_files,
            "conversation_id": conversation_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload files: {str(e)}")

@app.post("/api/ask-me-delete-context")
async def delete_ask_me_context(request: Request):
    """Delete cached context for a specific conversation"""
    try:
        data = await request.json()
        conversation_id = data.get("conversation_id", "")
        
        if not conversation_id:
            raise HTTPException(status_code=400, detail="Conversation ID is required")
        
        delete_ask_me_context_cache(conversation_id)
        
        return {"message": f"Context cache deleted for conversation {conversation_id}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete context: {str(e)}")

@app.post("/api/deep-analysis")
async def start_deep_analysis(background_tasks: BackgroundTasks, request: Request):
    """Start the deep analysis process"""
    try:
        # Parse and validate request data
        try:
            data = await request.json()
        except Exception as e:
            logger.error(f"Invalid JSON in deep analysis request: {e}")
            raise HTTPException(status_code=400, detail="Invalid JSON data")
        
        selected_caches = data.get("selected_caches", [])
        statements_file = data.get("statements_file", "")
        selected_statements = data.get("selected_statements", [])
        
        # Validate inputs
        if not selected_caches:
            logger.warning("No knowledge sources selected for deep analysis")
            raise HTTPException(status_code=400, detail="No knowledge sources selected")
        
        if not statements_file and not selected_statements:
            logger.warning("No statements file or statements selected for deep analysis")
            raise HTTPException(status_code=400, detail="No statements to process")
        
        # Check if statements file exists
        if statements_file:
            statements_path = UPLOAD_DIR / statements_file
            if not statements_path.exists():
                logger.error(f"Statements file not found for deep analysis: {statements_file}")
                raise HTTPException(status_code=404, detail=f"Statements file '{statements_file}' not found")
        
        logger.info(f"Starting deep analysis with {len(selected_caches)} caches")
        
        # Reset deep analysis status
        global deep_analysis_status
        deep_analysis_status = {
            "status": "idle",
            "progress": 0,
            "current_step": "",
            "processed_items": 0,
            "total_items": 0,
            "message": "",
            "logs": []
        }
        
        # Start background deep analysis processing
        background_tasks.add_task(
            process_deep_analysis_background, 
            selected_caches, 
            statements_file,
            selected_statements
        )
        
        return {"message": "Deep analysis started successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting deep analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to start deep analysis")

@app.get("/api/deep-analysis-status")
async def get_deep_analysis_status():
    """Get current deep analysis processing status"""
    global deep_analysis_status
    return deep_analysis_status

@app.get("/api/deep-results")
async def get_deep_results():
    """Get deep analysis results as JSON for display"""
    try:
        results_file = Path("Deep Verification results.xlsx")

        if not results_file.exists():
            logger.warning("Deep analysis results file not found")
            raise HTTPException(status_code=404, detail="Deep analysis results not found. Please run deep analysis first.")

        # Read the Excel file and convert to JSON
        import pandas as pd
        df = pd.read_excel(results_file)
        results = df.to_dict('records')
        
        logger.info(f"Serving deep analysis results as JSON: {len(results)} records")
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving deep analysis results: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve deep analysis results")

@app.get("/api/deep-results/download")
async def download_deep_results():
    """Download deep analysis results as Excel file"""
    try:
        results_file = Path("Deep Verification results.xlsx")

        if not results_file.exists():
            logger.warning("Deep analysis results file not found for download")
            raise HTTPException(status_code=404, detail="Deep analysis results not found. Please run deep analysis first.")

        logger.info(f"Serving deep analysis results file for download: {results_file}")
        return FileResponse(
            path=str(results_file),
            filename="Deep Verification results.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving deep analysis results for download: {e}")
        raise HTTPException(status_code=500, detail="Failed to serve deep analysis results")





async def process_deep_analysis_background(selected_caches, statements_file, selected_statements):
    """Background task for deep analysis processing"""
    try:
        update_deep_analysis_status("processing", 0, "Initializing deep analysis...", 0, 0)
        
        # Load API credentials
        api_key, base_url = load_api_credentials("api.txt")
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        update_deep_analysis_status("processing", 10, "Loading knowledge base...", 0, 0, 
                                   log="Loading knowledge base from selected caches")
        
        # Load embeddings and chunks from selected caches
        embeddings, chunks, cache_names = load_multiple_caches(selected_caches)
        update_deep_analysis_status("processing", 20, "Knowledge base loaded", 0, 0, 
                                   log=f"Loaded {len(chunks)} chunks from {len(selected_caches)} caches")
        
        # Read statements
        statements_path = UPLOAD_DIR / statements_file
        statements = read_statements(str(statements_path))
        
        if selected_statements:
            statements = [statements[i] for i in selected_statements if i < len(statements)]
        
        update_deep_analysis_status("processing", 30, f"Processing {len(statements)} statements...", 0, len(statements))
        
                # Process statements with deep analysis using parallel processing
        update_deep_analysis_status("processing", 30, f"Starting parallel processing of {len(statements)} statements...", 0, len(statements))
        
        # Create a function for processing individual statements
        def process_single_statement(statement_data):
            statement, index = statement_data
            try:
                # Get relevant chunks for this statement
                statement_embedding = embed_single_text(client, "text-embedding-3-large", statement['content'])
                
                # Calculate similarities
                similarities = []
                for j, embedding in enumerate(embeddings):
                    similarity_matrix = cosine_similarity([statement_embedding], [embedding])
                    similarity = float(similarity_matrix[0][0])  # Convert to scalar float
                    similarities.append((similarity, j))
                
                # Get top relevant chunks
                similarities.sort(reverse=True)
                top_chunks = []
                for similarity, idx in similarities[:15]:  # More chunks for deep analysis
                    if similarity > 0.1:
                        chunk_text = chunks[idx] if isinstance(chunks[idx], str) else chunks[idx][1]
                        top_chunks.append(chunk_text)
                
                # Create context from relevant chunks
                context = "\n\n".join(top_chunks[:15])
                if len(context) > 40000:
                    context = context[:40000] + "..."
                
                # Deep analysis prompt
                prompt = f"""Act as an experienced UK solicitor. Analyse the provided statement against the specified knowledge base. Determine whether the statement corresponds to the knowledge base. If it does not, explain why, citing the relevant documents or provisions. Propose how the statement could be appealed, setting out the strongest legal grounds for challenge. Limit your response to 150 words.

Knowledge Base Context:
{context}

Statement to Analyze:
{statement['content']}

Please provide your analysis in the following JSON format:
{{
    "legal_analysis": "your legal analysis here",
    "appeal_grounds": "your appeal grounds here",
    "degree_of_accuracy": "integer 1-10, where 10 is completely accurate"
}}"""
                
                # Get deep analysis from the model
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=[
                        {"role": "system", "content": "You are an experienced UK solicitor providing legal analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    max_completion_tokens=6000,
                    reasoning_effort="high",
                    verbosity="medium"
                )
                
                if response.choices and len(response.choices) > 0:
                    analysis_text = response.choices[0].message.content.strip()
                    
                    # Try to parse JSON response
                    try:
                        import json
                        analysis_data = json.loads(analysis_text)
                        legal_analysis = analysis_data.get("legal_analysis", analysis_text)
                        appeal_grounds = analysis_data.get("appeal_grounds", "")
                        degree_of_accuracy = analysis_data.get("degree_of_accuracy", 5)
                        
                        # Ensure degree_of_accuracy is a valid integer between 1-10
                        try:
                            degree_of_accuracy = int(degree_of_accuracy)
                            if degree_of_accuracy < 1:
                                degree_of_accuracy = 1
                            elif degree_of_accuracy > 10:
                                degree_of_accuracy = 10
                        except (ValueError, TypeError):
                            degree_of_accuracy = 5  # Default if parsing fails
                    except json.JSONDecodeError:
                        legal_analysis = analysis_text
                        appeal_grounds = ""
                        degree_of_accuracy = 5
                    
                    result = {
                        'Analysed Par number': statement['par'],
                        'Par Context': statement['content'],
                        'Legal Analysis': legal_analysis,
                        'Appeal Grounds': appeal_grounds,
                        'Degree of Accuracy': degree_of_accuracy,
                        'index': index
                    }
                    
                    update_deep_analysis_status("processing", 30 + ((index + 1) / len(statements)) * 60, 
                                              f"Completed statement {index + 1}/{len(statements)}", index + 1, len(statements),
                                              log=f"Completed deep analysis for statement {statement['par']}")
                    
                    return result
                else:
                    result = {
                        'Analysed Par number': statement['par'],
                        'Par Context': statement['content'],
                        'Legal Analysis': 'Analysis failed',
                        'Appeal Grounds': 'No appeal grounds available',
                        'Degree of Accuracy': 0,
                        'index': index
                    }
                    
                    update_deep_analysis_status("processing", 30 + ((index + 1) / len(statements)) * 60, 
                                              f"Failed statement {index + 1}/{len(statements)}", index + 1, len(statements),
                                              log=f"Analysis failed for statement {statement['par']}")
                    
                    return result
                    
            except Exception as e:
                result = {
                    'Analysed Par number': statement['par'],
                    'Par Context': statement['content'],
                    'Legal Analysis': f'Error: {str(e)}',
                    'Appeal Grounds': 'Analysis failed',
                    'Degree of Accuracy': 0,
                    'index': index
                }
                
                update_deep_analysis_status("processing", 30 + ((index + 1) / len(statements)) * 60, 
                                          f"Error in statement {index + 1}/{len(statements)}", index + 1, len(statements),
                                          log=f"Error analyzing statement {statement['par']}: {str(e)}")
                
                return result
        
        # Process statements in parallel using ThreadPoolExecutor
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        max_workers = min(8, len(statements))  # Use up to 8 workers for deep analysis
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_statement = {
                executor.submit(process_single_statement, (statement, i)): i 
                for i, statement in enumerate(statements)
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_statement):
                result = future.result()
                results.append(result)
        
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x['index'])
        # Remove the index field from final results
        for result in results:
            result.pop('index', None)
        
        # Save results to Excel
        update_deep_analysis_status("processing", 90, "Saving results...", len(statements), len(statements))
        
        df = pd.DataFrame(results)
        output_file = "Deep Verification results.xlsx"
        df.to_excel(output_file, index=False)
        
        update_deep_analysis_status("completed", 100, "Deep analysis completed", len(statements), len(statements),
                                   log=f"Deep analysis completed. Results saved to {output_file}")
        
    except Exception as e:
        update_deep_analysis_status("error", 0, "Deep analysis failed", 0, 0, 
                                   log=f"Deep analysis failed: {str(e)}")
        logger.error(f"Deep analysis error: {e}")

# Global variables for Ask Me context file caching
ask_me_context_caches = {}  # conversation_id -> cache_data
ask_me_cache_counter = 0

def generate_conversation_id():
    """Generate a unique conversation ID for Ask Me context caching"""
    global ask_me_cache_counter
    ask_me_cache_counter += 1
    return f"ask_me_conv_{ask_me_cache_counter}_{int(time.time())}"

def create_ask_me_context_cache(conversation_id: str, files_data: list):
    """Create a cache for Ask Me context files with indexing and embedding"""
    try:
        # Load API credentials
        api_key, base_url = load_api_credentials("api.txt")
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        # Combine all file contents
        combined_text = "\n\n".join([file_data["content"] for file_data in files_data])
        
        # Create temporary file for processing
        temp_file_path = UPLOAD_DIR / f"temp_ask_me_{conversation_id}.txt"
        with open(temp_file_path, "w", encoding="utf-8") as f:
            f.write(combined_text)
        
        # Ensure cache directory exists
        cache_dir = ".ask_me_cache"
        os.makedirs(cache_dir, exist_ok=True)
        
        # Build index for the combined content
        embeddings, chunks, index_id = build_or_load_index(
            corpus_path=str(temp_file_path),
            embedding_model="text-embedding-3-large",
            client=client,
            chunk_size_chars=4000,
            overlap_chars=500,
            cache_dir=cache_dir  # Separate cache directory for Ask Me
        )
        
        # Store cache data
        cache_data = {
            "conversation_id": conversation_id,
            "embeddings": embeddings,
            "chunks": chunks,
            "index_id": index_id,
            "files_data": files_data,
            "created_at": time.time()
        }
        
        ask_me_context_caches[conversation_id] = cache_data
        
        # Clean up temporary file
        if temp_file_path.exists():
            temp_file_path.unlink()
        
        logger.info(f"Created Ask Me context cache for conversation {conversation_id} with {len(chunks)} chunks")
        return cache_data
        
    except Exception as e:
        logger.error(f"Error creating Ask Me context cache: {str(e)}")
        raise

def get_ask_me_context_cache(conversation_id: str):
    """Get cached context data for a conversation"""
    return ask_me_context_caches.get(conversation_id)

def delete_ask_me_context_cache(conversation_id: str):
    """Delete cached context data for a conversation"""
    try:
        if conversation_id in ask_me_context_caches:
            cache_data = ask_me_context_caches[conversation_id]
            
            # Delete cache files from disk
            cache_dir = ".ask_me_cache"
            index_id = cache_data["index_id"]
            
            emb_path = os.path.join(cache_dir, f"{index_id}.npz")
            chunks_path = os.path.join(cache_dir, f"{index_id}.chunks.jsonl")
            
            if os.path.exists(emb_path):
                os.remove(emb_path)
            if os.path.exists(chunks_path):
                os.remove(chunks_path)
            
            # Remove from memory
            del ask_me_context_caches[conversation_id]
            
            logger.info(f"Deleted Ask Me context cache for conversation {conversation_id}")
            
    except Exception as e:
        logger.error(f"Error deleting Ask Me context cache: {str(e)}")

def cleanup_expired_ask_me_caches(max_age_hours: int = 24):
    """Clean up expired Ask Me context caches"""
    current_time = time.time()
    expired_conversations = []
    
    for conversation_id, cache_data in ask_me_context_caches.items():
        age_hours = (current_time - cache_data["created_at"]) / 3600
        if age_hours > max_age_hours:
            expired_conversations.append(conversation_id)
    
    for conversation_id in expired_conversations:
        delete_ask_me_context_cache(conversation_id)
    
    if expired_conversations:
        logger.info(f"Cleaned up {len(expired_conversations)} expired Ask Me context caches")

if __name__ == "__main__":
    logger.info("Starting VeriDoc AI Web Interface...")
    
    # Clean up expired Ask Me context caches on startup
    cleanup_expired_ask_me_caches()
    
    logger.info("Open your browser and go to: http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
