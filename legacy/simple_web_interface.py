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
from pathlib import Path
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
import pandas as pd

# Import your existing verification script
from verify_statements import (
    load_api_credentials, build_or_load_index, read_statements,
    process_statements_parallel, excel_prepare_court_writer,
    excel_append_row, sort_excel_by_paragraph_number,
    list_available_caches, delete_cache, load_multiple_caches,
    run as run_verification
)

app = FastAPI(title="Document Verification API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for processing status
processing_status = {
    "status": "idle",
    "progress": 0,
    "current_step": "",
    "processed_items": 0,
    "total_items": 0,
    "message": "",
    "logs": []
}

# File storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def update_status(status: str, progress: float = None, current_step: str = None, 
                 processed_items: int = None, total_items: int = None, 
                 message: str = None, log: str = None):
    """Update processing status"""
    global processing_status
    
    if status:
        processing_status["status"] = status
    if progress is not None:
        processing_status["progress"] = progress
    if current_step:
        processing_status["current_step"] = current_step
    if processed_items is not None:
        processing_status["processed_items"] = processed_items
    if total_items is not None:
        processing_status["total_items"] = total_items
    if message:
        processing_status["message"] = message
    if log:
        processing_status["logs"].append(f"[{time.strftime('%H:%M:%S')}] {log}")
        if len(processing_status["logs"]) > 100:
            processing_status["logs"] = processing_status["logs"][-100:]

async def process_verification_background(source_files: list, statements_file: str):
    """Background task for processing verification"""
    try:
        update_status("processing", 0, "Initializing...", 0, 0, "Starting verification process")
        
        # Load API credentials
        update_status("processing", 5, "Loading API credentials...", 0, 0)
        api_key, base_url = load_api_credentials("api.txt")
        update_status("processing", 10, "API credentials loaded", 0, 0)
        
        # Load statements
        update_status("processing", 15, "Loading statements file...", 0, 0)
        statements_path = UPLOAD_DIR / statements_file
        statements = read_statements(str(statements_path))
        update_status("processing", 20, f"Loaded {len(statements)} statements", 0, len(statements))
        
        # Process source files
        update_status("processing", 25, "Processing source files...", 0, len(statements))
        main_source_file = UPLOAD_DIR / source_files[0]
        
        # Build or load index
        update_status("processing", 30, "Building/loading document index...", 0, len(statements))
        from openai import OpenAI
        client = OpenAI(api_key=api_key, base_url=base_url)
        
        embeddings, chunks, index_id = build_or_load_index(
            corpus_path=str(main_source_file),
            embedding_model="text-embedding-3-large",
            client=client,
            chunk_size_chars=4000,
            overlap_chars=500,
        )
        update_status("processing", 50, "Document index ready", 0, len(statements))
        
        # Process statements using the real verification logic
        update_status("processing", 60, "Processing statements...", 0, len(statements))
        
        # Use the actual verification logic from your script
        results = []
        for i, statement in enumerate(statements):
            try:
                # Get the three-line context
                par_number = statement['par']
                content = statement['content']
                
                # Create three-line context (current + previous + next)
                three_line_content = content
                if i > 0:
                    three_line_content = statements[i-1]['content'] + "\n" + three_line_content
                if i < len(statements) - 1:
                    three_line_content = three_line_content + "\n" + statements[i+1]['content']
                
                # Get relevant evidence chunks
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as np
                
                # Get embedding for the statement
                statement_embedding = client.embeddings.create(
                    input=three_line_content,
                    model="text-embedding-3-large"
                ).data[0].embedding
                
                # Find most similar chunks
                similarities = cosine_similarity([statement_embedding], embeddings)[0]
                top_indices = np.argsort(similarities)[-10:][::-1]  # Top 10
                
                evidence_snippets = []
                for idx in top_indices:
                    if similarities[idx] > 0.1:  # Only include relevant chunks
                        evidence_snippets.append(chunks[idx][:800])  # Limit snippet size
                
                # Judge the statement
                from verify_statements import build_court_verification_prompt, judge_court_statement
                
                prompt = build_court_verification_prompt(par_number, three_line_content, evidence_snippets)
                
                response = client.chat.completions.create(
                    model="gpt-5-mini",
                    messages=prompt,
                    temperature=0.1,
                    max_tokens=500
                )
                
                result = judge_court_statement(response.choices[0].message.content)
                result['par_number'] = par_number
                result['par_context'] = three_line_content
                
                results.append(result)
                
                # Update progress
                progress = 60 + (i / len(statements)) * 30
                update_status("processing", progress, f"Processing statement {i+1}/{len(statements)}", i+1, len(statements))
                
            except Exception as e:
                print(f"Error processing statement {i+1}: {e}")
                # Add error result
                results.append({
                    'par_number': par_number,
                    'par_context': three_line_content,
                    'is_accurate': 'error',
                    'degree_of_accuracy': 0,
                    'inaccuracy_type': 'error',
                    'description': f'Processing error: {str(e)}'
                })
        
        # Save results to Excel
        update_status("processing", 90, "Saving results...", len(statements), len(statements))
        
        # Create Excel file with results
        excel_path, workbook = excel_prepare_court_writer("verification_results.xlsx")
        
        for result in results:
            excel_append_row(workbook, excel_path, [
                result['par_number'],
                result['par_context'],
                result['is_accurate'],
                result['degree_of_accuracy'],
                result['inaccuracy_type'],
                result['description']
            ])
        
        # Sort the Excel file
        sort_excel_by_paragraph_number("verification_results.xlsx")
        
        update_status("completed", 100, "Processing completed", len(statements), len(statements), 
                     "Verification process completed successfully")
        
    except Exception as e:
        update_status("error", 0, "Error occurred", 0, 0, f"Error: {str(e)}")
        print(f"Processing error: {e}")

@app.get("/", response_class=HTMLResponse)
async def get_main_page():
    """Serve the main HTML page"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Verification System</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        .header {
            background: #2563eb;
            color: white;
            padding: 20px;
            text-align: center;
        }
        .content {
            padding: 20px;
        }
        .tab-container {
            display: flex;
            border-bottom: 1px solid #ddd;
            margin-bottom: 20px;
        }
        .tab {
            padding: 10px 20px;
            cursor: pointer;
            border: none;
            background: none;
            font-size: 16px;
        }
        .tab.active {
            background: #2563eb;
            color: white;
            border-radius: 4px 4px 0 0;
        }
        .tab-content {
            display: none;
        }
        .tab-content.active {
            display: block;
        }
        .upload-area {
            border: 2px dashed #ccc;
            border-radius: 8px;
            padding: 40px;
            text-align: center;
            margin: 20px 0;
            cursor: pointer;
        }
        .upload-area:hover {
            border-color: #2563eb;
        }
        .file-list {
            margin: 20px 0;
        }
        .file-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border: 1px solid #ddd;
            margin: 5px 0;
            border-radius: 4px;
        }
        .btn {
            background: #2563eb;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
        }
        .btn:hover {
            background: #1d4ed8;
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background: #2563eb;
            transition: width 0.3s ease;
        }
        .status {
            padding: 10px;
            margin: 10px 0;
            border-radius: 4px;
        }
        .status.processing {
            background: #dbeafe;
            color: #1e40af;
        }
        .status.completed {
            background: #dcfce7;
            color: #166534;
        }
        .status.error {
            background: #fee2e2;
            color: #991b1b;
        }
        .results-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .results-table th,
        .results-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .results-table th {
            background: #f8f9fa;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>Document Verification System</h1>
            <p>AI-Powered Fact Checking</p>
        </div>
        
        <div class="content">
            <div class="tab-container">
                <button class="tab active" data-tab="upload">Upload Files</button>
                <button class="tab" data-tab="caches">Cache Manager</button>
                <button class="tab" data-tab="processing">Processing</button>
                <button class="tab" data-tab="results">Results</button>
            </div>
            
            <div id="upload" class="tab-content active">
                <h2>Upload Files</h2>
                <p>Upload your source documents and statements to verify</p>
                
                <div class="upload-area" id="uploadArea">
                    <p>Click to select files or drag and drop</p>
                    <p>Supports: TXT, PDF, DOCX, XLSX</p>
                    <input type="file" id="fileInput" multiple style="display: none;">
                </div>
                
                <div id="fileList" class="file-list"></div>
                
                <button id="startBtn" class="btn" disabled>Start Verification Process</button>
            </div>
            
            <div id="processing" class="tab-content">
                <h2>Processing Status</h2>
                <div id="statusDisplay">
                    <p>No active processing</p>
                </div>
                <div class="progress-bar">
                    <div id="progressFill" class="progress-fill" style="width: 0%"></div>
                </div>
                <div id="logs"></div>
            </div>
            
            <div id="caches" class="tab-content">
                <h2>Cache Manager</h2>
                <p>Manage your cached source files for faster processing</p>
                <button class="btn" id="loadCachesBtn">Load Caches</button>
                <button class="btn" id="deleteCachesBtn" style="background: #dc2626;">Delete Selected</button>
                <div id="cachesTable"></div>
            </div>
            
            <div id="results" class="tab-content">
                <h2>Results</h2>
                <button class="btn" id="loadResultsBtn">Load Results</button>
                <button class="btn" id="downloadResultsBtn">Download Results</button>
                <div id="resultsTable"></div>
            </div>
        </div>
    </div>
    
    <script>
        let uploadedFiles = [];
        let selectedCaches = [];
        let statusPolling = null;
        
        // Initialize when DOM is loaded
        document.addEventListener('DOMContentLoaded', function() {
            initializeEventListeners();
        });
        
        function initializeEventListeners() {
            // Tab switching
            const tabButtons = document.querySelectorAll('.tab-container .tab');
            tabButtons.forEach(btn => {
                btn.addEventListener('click', function() {
                    const tabName = this.getAttribute('data-tab');
                    showTab(tabName);
                });
            });
            
            // File upload
            const uploadArea = document.getElementById('uploadArea');
            const fileInput = document.getElementById('fileInput');
            if (uploadArea && fileInput) {
                uploadArea.addEventListener('click', function() {
                    fileInput.click();
                });
                fileInput.addEventListener('change', handleFileSelect);
            }
            
            // Buttons
            const startBtn = document.getElementById('startBtn');
            if (startBtn) {
                startBtn.addEventListener('click', startProcessing);
            }
            
            const loadCachesBtn = document.getElementById('loadCachesBtn');
            if (loadCachesBtn) {
                loadCachesBtn.addEventListener('click', loadCaches);
            }
            
            const deleteCachesBtn = document.getElementById('deleteCachesBtn');
            if (deleteCachesBtn) {
                deleteCachesBtn.addEventListener('click', deleteSelectedCaches);
            }
            
            const loadResultsBtn = document.getElementById('loadResultsBtn');
            if (loadResultsBtn) {
                loadResultsBtn.addEventListener('click', loadResults);
            }
            
            const downloadResultsBtn = document.getElementById('downloadResultsBtn');
            if (downloadResultsBtn) {
                downloadResultsBtn.addEventListener('click', downloadResults);
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
            } else {
                stopStatusPolling();
            }
        }
        
        function handleFileSelect(event) {
            const files = event.target.files;
            for (let file of files) {
                uploadedFiles.push({
                    name: file.name,
                    size: file.size,
                    type: file.name.toLowerCase().includes('judgement') ? 'statements' : 'source'
                });
            }
            updateFileList();
            uploadFiles(files);
        }
        
        function updateFileList() {
            const fileList = document.getElementById('fileList');
            fileList.innerHTML = '';
            
            uploadedFiles.forEach((file, index) => {
                const fileItem = document.createElement('div');
                fileItem.className = 'file-item';
                
                const fileSize = (file.size / 1024 / 1024).toFixed(2);
                const fileInfo = document.createElement('span');
                fileInfo.textContent = file.name + ' (' + fileSize + ' MB) - ' + file.type;
                
                const removeButton = document.createElement('button');
                removeButton.textContent = 'Remove';
                removeButton.style.cssText = 'background: #dc2626; color: white; border: none; padding: 5px 10px; border-radius: 4px; cursor: pointer;';
                removeButton.addEventListener('click', function() {
                    removeFileByIndex(index);
                });
                
                fileItem.appendChild(fileInfo);
                fileItem.appendChild(removeButton);
                fileList.appendChild(fileItem);
            });
            
            // Update start button state
            const hasSource = uploadedFiles.some(f => f.type === 'source');
            const hasStatements = uploadedFiles.some(f => f.type === 'statements');
            const startBtn = document.getElementById('startBtn');
            if (startBtn) {
                startBtn.disabled = !(hasSource && hasStatements);
            }
        }
        
        function removeFileByIndex(index) {
            if (index >= 0 && index < uploadedFiles.length) {
                uploadedFiles.splice(index, 1);
                updateFileList();
            }
        }
        
        async function uploadFiles(files) {
            for (let file of files) {
                const formData = new FormData();
                formData.append('file', file);
                
                try {
                    const response = await fetch('/api/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error('Upload failed');
                    }
                    
                    console.log(file.name + ' uploaded successfully');
                } catch (error) {
                    console.error('Failed to upload ' + file.name + ':', error);
                }
            }
        }
        
        async function startProcessing() {
            const sourceFiles = uploadedFiles.filter(f => f.type === 'source').map(f => f.name);
            const statementsEntry = uploadedFiles.find(f => f.type === 'statements');
            const statementsFile = statementsEntry ? statementsEntry.name : null;
            
            if (!sourceFiles.length || !statementsFile) {
                alert('Please upload at least one source file and one statements file');
                return;
            }
            
            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        source_files: sourceFiles,
                        statements_file: statementsFile
                    })
                });
                
                if (!response.ok) {
                    throw new Error('Failed to start processing');
                }
                
                showTab('processing');
            } catch (error) {
                console.error('Error starting processing:', error);
                alert('Failed to start processing');
            }
        }
        
        function startStatusPolling() {
            statusPolling = setInterval(async () => {
                try {
                    const response = await fetch('/api/status');
                    const status = await response.json();
                    updateStatusDisplay(status);
                    
                    if (status.status === 'completed' || status.status === 'error') {
                        stopStatusPolling();
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
            
            let tableHTML = '<table class="results-table"><thead><tr>';
            const headers = Object.keys(results[0]);
            headers.forEach(header => {
                tableHTML += '<th>' + header + '</th>';
            });
            tableHTML += '</tr></thead><tbody>';
            
            results.forEach(result => {
                tableHTML += '<tr>';
                headers.forEach(header => {
                    tableHTML += '<td>' + (result[header] || '') + '</td>';
                });
                tableHTML += '</tr>';
            });
            
            tableHTML += '</tbody></table>';
            resultsTable.innerHTML = tableHTML;
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
        
        async function loadCaches() {
            try {
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
                cachesTable.innerHTML = '<p>No cached files found</p>';
                return;
            }
            
            let tableHTML = '<table class="results-table"><thead><tr><th><input type="checkbox" id="selectAllCaches"></th><th>Cache ID</th><th>Original Name</th><th>Total Size (MB)</th><th>NPZ Size (MB)</th><th>Chunks Size (MB)</th></tr></thead><tbody>';
            
            caches.forEach((cache, index) => {
                tableHTML += '<tr><td><input type="checkbox" class="cache-checkbox" value="' + cache.cache_id + '"></td><td>' + 
                    cache.cache_id + '</td><td>' + cache.original_name + '</td><td>' + 
                    cache.total_size_mb.toFixed(1) + '</td><td>' + cache.npz_size_mb.toFixed(1) + '</td><td>' + 
                    cache.chunks_size_mb.toFixed(1) + '</td></tr>';
            });
            
            tableHTML += '</tbody></table>';
            cachesTable.innerHTML = tableHTML;
            
            // Add event listeners for checkboxes
            const selectAllCheckbox = document.getElementById('selectAllCaches');
            if (selectAllCheckbox) {
                selectAllCheckbox.addEventListener('change', function() {
                    const checkboxes = document.querySelectorAll('.cache-checkbox');
                    checkboxes.forEach(cb => {
                        cb.checked = this.checked;
                        if (this.checked) {
                            if (!selectedCaches.includes(cb.value)) {
                                selectedCaches.push(cb.value);
                            }
                        } else {
                            selectedCaches = selectedCaches.filter(id => id !== cb.value);
                        }
                    });
                });
            }
            
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
                });
            });
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
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(selectedCaches)
                });
                
                if (!response.ok) {
                    throw new Error('Failed to delete caches');
                }
                
                alert('Caches deleted successfully');
                selectedCaches = [];
                loadCaches();
            } catch (error) {
                console.error('Error deleting caches:', error);
                alert('Failed to delete caches');
            }
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file"""
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {"message": "File uploaded successfully", "filename": file.filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@app.post("/api/process")
async def start_processing(background_tasks: BackgroundTasks, request: Request):
    """Start the verification process"""
    try:
        data = await request.json()
        source_files = data.get("source_files", [])
        statements_file = data.get("statements_file", "")
        
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
            source_files, 
            statements_file
        )
        
        return {"message": "Processing started successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start processing: {str(e)}")

@app.get("/api/caches")
async def get_caches():
    """Get list of available caches"""
    try:
        caches = list_available_caches()
        return caches
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch caches: {str(e)}")

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
        return df.to_dict('records')
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

@app.get("/api/status")
async def get_status():
    """Get current processing status"""
    return processing_status

if __name__ == "__main__":
    print("Starting Document Verification Web Interface...")
    print("Open your browser and go to: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
