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
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
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

# Suppress asyncio connection reset errors
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="asyncio")

# Configure asyncio to handle connection errors gracefully
import asyncio
import signal
import sys

def handle_connection_error():
    """Handle connection reset errors gracefully"""
    try:
        # Suppress the specific connection reset error
        pass
    except Exception:
        pass

# Set up signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

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
    update_status, get_processing_status,
    reset_processing_status, processing_status
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

# Add exception handler for connection errors
@app.exception_handler(ConnectionResetError)
async def connection_reset_handler(request: Request, exc: ConnectionResetError):
    """Handle connection reset errors gracefully"""
    logger.warning(f"Connection reset error: {exc}")
    return {"detail": "Connection was reset"}

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return {"detail": "Internal server error"}

# Startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    """Handle application startup"""
    logger.info("🚀 VeriDoc AI application starting up...")
    
    # Suppress asyncio connection reset errors
    import warnings
    warnings.filterwarnings("ignore", message=".*ConnectionResetError.*")
    warnings.filterwarnings("ignore", message=".*WinError 10054.*")
    
    logger.info("✅ Application startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown"""
    logger.info("🛑 VeriDoc AI application shutting down...")
    logger.info("✅ Application shutdown completed")



# Mount static files
app.mount("/static", StaticFiles(directory="."), name="static")

# Global variables for uploaded files
knowledge_files = []
statements_file = None

# Server-side category storage
cache_categories = {}
all_categories = []
CATEGORIES_FILE = Path("config/cache_categories.json")
CATEGORY_LIST_FILE = Path("config/category_list.json")

# Project management storage
projects = {}
current_project = None
PROJECTS_FILE = Path("config/projects.json")
PROJECT_CATEGORIES_FILE = Path("config/project_categories.json")

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
        # Load project-specific categories if a project is selected
        if current_project:
            cache_categories, all_categories = load_project_categories(current_project)
            logger.info(f"Loaded project-specific categories for '{current_project}': {len(cache_categories)} cache categories and {len(all_categories)} categories")
        else:
            # Fallback to global categories for backward compatibility
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

def switch_to_project_categories(project_name: str):
    """Switch to project-specific categories"""
    global cache_categories, all_categories
    try:
        # Clear global categories first to prevent contamination
        cache_categories = {}
        all_categories = DEFAULT_CATEGORIES.copy()
        
        # Load project-specific categories
        project_categories, project_all_categories = load_project_categories(project_name)
        cache_categories = project_categories
        all_categories = project_all_categories
        
        logger.info(f"Switched to project-specific categories for '{project_name}': {len(cache_categories)} cache categories and {len(all_categories)} categories")
    except Exception as e:
        logger.error(f"Error switching to project categories for {project_name}: {e}")
        cache_categories = {}
        all_categories = DEFAULT_CATEGORIES.copy()

def save_cache_categories():
    """Save cache categories to file"""
    try:
        # Save to project-specific file if a project is selected
        if current_project:
            save_project_categories(current_project, cache_categories, all_categories)
            logger.info(f"Saved {len(cache_categories)} cache categories to project: {current_project}")
        else:
            # Save to global file for backward compatibility
            with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
                json.dump(cache_categories, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(cache_categories)} cache categories to file")
        
        # Save to category list file
        save_category_list()
        
    except Exception as e:
        logger.error(f"Error saving cache categories: {e}")

def save_category_list():
    """Save category list to file"""
    try:
        # Save to global file for backward compatibility
        with open(CATEGORY_LIST_FILE, 'w', encoding='utf-8') as f:
            json.dump(all_categories, f, ensure_ascii=False, indent=2)
        
        # Save to current project's categories file
        if current_project:
            save_project_categories(current_project, cache_categories, all_categories)
        
        logger.info(f"Saved {len(all_categories)} categories to category list")
    except Exception as e:
        logger.error(f"Error saving category list: {e}")

# Load projects first, then categories
# Note: load_projects() and load_cache_categories() are called after function definitions

# File storage
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
        cache_dir = get_project_ask_me_cache_dir(current_project) if current_project else ".ask_me_cache"
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
            cache_dir = get_project_ask_me_cache_dir(current_project) if current_project else ".ask_me_cache"
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

def process_statement_batch(client, judge_model: str, embedding_model: str, 
                           statements_batch: list, embeddings: np.ndarray, 
                           chunks: list, top_k: int = 10, max_snippet_chars: int = 4000,
                           role: str = "You are an experienced UK solicitor providing legal analysis.",
                           max_completion_tokens: int = 4000, reasoning_effort: str = "medium", 
                           verbosity: str = "low", output_fields: dict = None, output_field_descriptions: dict = None) -> list:
    """Process a batch of statements in parallel"""
    results = []
    
    # Prepare statements for embedding
    statements_texts = []
    for stmt in statements_batch:
        # Create three-line context
        content = stmt['content']
        three_line_content = content
        statements_texts.append(three_line_content)
    
    # Batch embed all statements using utility function
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
            
            # Judge the statement with custom parameters
            result = judge_court_statement_with_params(client, judge_model, stmt['par'], statements_texts[i], evidence_snippets,
                                                      role=role, max_completion_tokens=max_completion_tokens, 
                                                      reasoning_effort=reasoning_effort, verbosity=verbosity,
                                                      output_field_descriptions=output_field_descriptions)
            
            # Prepare result based on output field selections
            result_dict = {}
            
            # Use output_fields if provided, otherwise include all fields
            if output_fields is None:
                output_fields = {
                    'par_number': True,
                    'par_context': True,
                    'is_accurate': True,
                    'field_1756803554789': True,
                    'field_1756803586927': True
                }
            
            # Get field names from output_field_descriptions if available
            field_names = {}
            if output_field_descriptions:
                # Load the current field configuration to get names
                try:
                    output_fields_file = "config/output_fields_config.json"
                    if os.path.exists(output_fields_file):
                        with open(output_fields_file, 'r', encoding='utf-8') as f:
                            field_config = json.load(f)
                            for field in field_config:
                                field_names[field['id']] = field['name']
                except Exception as e:
                    logger.error(f"Error loading field names: {e}")
            
            # Map field IDs to their corresponding data with custom names
            field_mapping = {
                'par_number': (field_names.get('par_number', 'Paragraph Number'), stmt['par']),
                'par_context': (field_names.get('par_context', 'Statement'), statements_texts[i]),
                'is_accurate': (field_names.get('is_accurate', 'Document Reference'), result.get('is_accurate', 'N/A')),
                'field_1756803554789': (field_names.get('field_1756803554789', 'Fact/Finding'), result.get('field_1756803554789', 'N/A')),
                'field_1756803586927': (field_names.get('field_1756803586927', 'Findings'), result.get('field_1756803586927', 'N/A'))
            }
            
            # Add enabled fields to result
            for field_id, enabled in output_fields.items():
                if enabled and field_id in field_mapping:
                    column_name, value = field_mapping[field_id]
                    result_dict[column_name] = value
                elif enabled and field_id.startswith('field_'):
                    # Handle custom fields from AI response
                    custom_value = result.get(field_id, 'Custom field - see description')
                    result_dict[field_id] = custom_value
            
            results.append(result_dict)
            
        except Exception as e:
            # Handle errors gracefully
            result_dict = {}
            
            # Use output_fields if provided, otherwise include all fields
            if output_fields is None:
                output_fields = {
                    'par_number': True,
                    'par_context': True,
                    'is_accurate': True,
                    'field_1756803554789': True,
                    'field_1756803586927': True
                }
            
            # Get field names from output_field_descriptions if available for error case
            field_names = {}
            if output_field_descriptions:
                # Load the current field configuration to get names
                try:
                    output_fields_file = "config/output_fields_config.json"
                    if os.path.exists(output_fields_file):
                        with open(output_fields_file, 'r', encoding='utf-8') as f:
                            field_config = json.load(f)
                            for field in field_config:
                                field_names[field['id']] = field['name']
                except Exception as e:
                    logger.error(f"Error loading field names: {e}")
            
            # Map field IDs to their corresponding data for error case with custom names
            field_mapping = {
                'par_number': (field_names.get('par_number', 'Paragraph Number'), stmt['par']),
                'par_context': (field_names.get('par_context', 'Statement'), stmt['content']),
                'is_accurate': (field_names.get('is_accurate', 'Document Reference'), 'error'),
                'field_1756803554789': (field_names.get('field_1756803554789', 'Fact/Finding'), 'error'),
                'field_1756803586927': (field_names.get('field_1756803586927', 'Findings'), 'error')
            }
            
            # Add enabled fields to result
            for field_id, enabled in output_fields.items():
                if enabled and field_id in field_mapping:
                    column_name, value = field_mapping[field_id]
                    result_dict[column_name] = value
                elif enabled and field_id.startswith('field_'):
                    # Handle custom fields in error case
                    result_dict[field_id] = f'Error: {str(e)}'
            
            results.append(result_dict)
    
    return results


def judge_court_statement_with_params(client, model: str, par_number: str, three_line_content: str, evidence_snippets: list,
                                     role: str = "You are an experienced UK solicitor providing legal analysis.",
                                     max_completion_tokens: int = 4000, reasoning_effort: str = "medium", 
                                     verbosity: str = "low", output_field_descriptions: dict = None) -> dict:
    """Judge a court statement with custom parameters"""
    try:
        # Build evidence context
        evidence_context = "\n\n".join(evidence_snippets[:10])  # Limit to top 10 snippets
        if len(evidence_context) > 40000:
            evidence_context = evidence_context[:40000] + "..."
        
        # Create the prompt with custom output field descriptions
        if output_field_descriptions:
            # Build custom field descriptions for the prompt
            field_descriptions = []
            custom_fields = {}
            
            # Create a more detailed prompt that uses the custom descriptions
            analysis_instructions = []
            
            for field_id, description in output_field_descriptions.items():
                if field_id == 'is_accurate':
                    field_descriptions.append(f'"verdict": "{description}"')
                    analysis_instructions.append(f"• {description}")
                elif field_id == 'degree_accuracy':
                    field_descriptions.append(f'"degree_of_accuracy": "{description}"')
                    analysis_instructions.append(f"• {description}")
                elif field_id == 'inaccuracy_type':
                    field_descriptions.append(f'"inaccuracy_type": "{description}"')
                    analysis_instructions.append(f"• {description}")
                elif field_id == 'description':
                    field_descriptions.append(f'"description": "{description}"')
                    analysis_instructions.append(f"• {description}")
                elif field_id.startswith('field_'):
                    # Handle custom fields
                    field_descriptions.append(f'"{field_id}": "{description}"')
                    custom_fields[field_id] = description
                    analysis_instructions.append(f"• {description}")
            
            custom_format = "{\n    " + ",\n    ".join(field_descriptions) + "\n}"
            analysis_instructions_text = "\n".join(analysis_instructions)
        else:
            # Default format
            custom_format = """{
    "verdict": "accurate" or "not accurate",
    "degree_of_accuracy": integer 1-10 (where 10 is completely accurate),
    "inaccuracy_type": "manipulation", "omission", "addition", or "none",
    "description": "detailed explanation of your analysis"
}"""
            analysis_instructions_text = """• Whether the statement is accurate or not accurate
• Level of accuracy (integer 1-10, where 10 is completely accurate)
• Type of inaccuracy if statement is inaccurate (manipulation, omission, addition, or none)
• Detailed explanation of your analysis"""

        prompt = f"""Act as {role}

Analyze the following statement against the provided knowledge base evidence and answer the specific questions below.

Knowledge Base Evidence:
{evidence_context}

Statement to Verify:
{three_line_content}

Please analyze the statement and answer these specific questions:
{analysis_instructions_text}

Provide your analysis in the following JSON format:
{custom_format}"""

        # Call the model
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_completion_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            response_format={"type": "json_object"},
            timeout=30
        )
        
        content = response.choices[0].message.content
        
        # Parse the response
        import json
        obj = json.loads(content)
        verdict = obj.get("verdict", "not accurate").strip().lower()
        degree_of_accuracy = obj.get("degree_of_accuracy", 5)
        inaccuracy_type = obj.get("inaccuracy_type", "none").strip().lower()
        description = obj.get("description", "")
        
        # Extract custom fields
        custom_field_values = {}
        if output_field_descriptions:
            for field_id in output_field_descriptions.keys():
                if field_id.startswith('field_'):
                    custom_field_values[field_id] = obj.get(field_id, "")
        
        # Validate verdict
        if verdict not in ("accurate", "not accurate"):
            verdict = "not accurate"
        
        # Validate inaccuracy type
        if inaccuracy_type not in ("manipulation", "omission", "addition", "none"):
            inaccuracy_type = "none"
        
        # Ensure degree_of_accuracy is a valid integer between 1-10
        try:
            degree_of_accuracy = int(degree_of_accuracy)
            if degree_of_accuracy < 1:
                degree_of_accuracy = 1
            elif degree_of_accuracy > 10:
                degree_of_accuracy = 10
        except (ValueError, TypeError):
            degree_of_accuracy = 5
        
        result = {
            "verdict": verdict,
            "degree_of_accuracy": degree_of_accuracy,
            "inaccuracy_type": inaccuracy_type if verdict == "not accurate" else "none",
            "description": description
        }
        
        # Add custom fields to result
        result.update(custom_field_values)
        
        return result
        
    except Exception as e:
        # Fallback response on error
        return {
            "verdict": "not accurate",
            "degree_of_accuracy": 3,
            "inaccuracy_type": "manipulation",
            "description": f"Error processing statement: {str(e)}"
        }


def process_statements_parallel(client, judge_model: str, embedding_model: str,
                               statements: list, embeddings: np.ndarray,
                               chunks: list, top_k: int = 10, max_snippet_chars: int = 4000,
                               max_workers: int = 8, role: str = "You are an experienced UK solicitor providing legal analysis.",
                               max_completion_tokens: int = 4000, reasoning_effort: str = "medium", 
                               verbosity: str = "low", output_fields: dict = None, output_field_descriptions: dict = None) -> list:
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
                embeddings, chunks, top_k, max_snippet_chars,
                role, max_completion_tokens, reasoning_effort, verbosity, output_fields, output_field_descriptions
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

async def process_verification_background(selected_caches: list, statements_filename: str, selected_statements: list = None, verification_params: dict = None, output_fields: dict = None, output_field_descriptions: dict = None):
    """Background task for processing verification"""
    from utils.status import update_status
    try:
        update_status("processing", 0, "Initializing...", 0, 0, "Starting quick analysis")
        
        # Load API credentials
        update_status("processing", 5, "Loading API credentials...", 0, 0)
        api_key, base_url = load_api_credentials("api.txt")
        update_status("processing", 10, "API credentials loaded", 0, 0)
        
        # Load statements
        update_status("processing", 15, "Loading statements file...", 0, 0)
        statements_path = UPLOAD_DIR / statements_filename
        
        try:
            all_statements = read_statements(str(statements_path))
        except Exception as e:
            error_msg = f"Failed to load statements file: {str(e)}"
            logger.error(error_msg)
            update_status("error", 0, error_msg, 0, 0, error_msg)
            return
        
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
            # Use existing caches with project-specific cache directory
            project_cache_dir = get_project_cache_dir(current_project) if current_project else ".cache"
            
            # Check if project-specific cache directory exists and migrate if needed
            if current_project and not project_cache_dir.exists():
                migrate_existing_caches_to_project(current_project)
            
            embeddings, chunks, cache_names = load_multiple_caches(selected_caches, str(project_cache_dir))
            update_status("processing", 50, "Knowledge database loaded from caches", 0, len(statements))
        else:
            # Process first knowledge file (for now, we'll use the first one)
            # In a full implementation, you'd combine multiple files
            main_source_file = UPLOAD_DIR / knowledge_files[0]['name']
            
            # Use project-specific cache directory
            project_cache_dir = get_project_cache_dir(current_project) if current_project else ".cache"
            os.makedirs(project_cache_dir, exist_ok=True)
            
            embeddings, chunks, index_id = build_or_load_index(
                corpus_path=str(main_source_file),
                embedding_model="text-embedding-3-large",
                client=client,
                chunk_size_chars=4000,
                overlap_chars=500,
                cache_dir=str(project_cache_dir)
            )
            update_status("processing", 50, "Knowledge database built", 0, len(statements))
        
        # Process statements using parallel processing
        update_status("processing", 60, "Starting parallel processing...", 0, len(statements))
        
        # Use parallel processing for better performance
        # Use verification parameters if provided, otherwise use defaults
        role = verification_params.get("role", "You are an experienced UK solicitor providing legal analysis.") if verification_params else "You are an experienced UK solicitor providing legal analysis."
        completion_tokens = verification_params.get("completion_tokens", 4000) if verification_params else 4000
        reasoning_effort = verification_params.get("reasoning_effort", "medium") if verification_params else "medium"
        verbosity = verification_params.get("verbosity", "low") if verification_params else "low"
        
        # Log processing parameters
        logger.info(f"Processing {len(statements)} statements with {len(selected_caches)} caches")
        
        results = process_statements_parallel(
            client=client,
            judge_model="gpt-5-mini",
            embedding_model="text-embedding-3-large",
            statements=statements,
            embeddings=embeddings,
            chunks=chunks,
            top_k=10,
            max_snippet_chars=4000,  # Use full chunk size
            max_workers=8,  # Adjust based on your system capabilities
            role=role,
            max_completion_tokens=completion_tokens,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
            output_fields=output_fields,
            output_field_descriptions=output_field_descriptions
        )
        
        # Save results to Excel with new structure
        update_status("processing", 90, "Saving results...", len(statements), len(statements))
        
        # Create Excel with fixed first two columns and dynamic output fields
        import openpyxl
        from openpyxl import Workbook
        import os
        import json
        
        # Create Excel with results
        logger.info(f"Creating Excel with {len(results)} results")
        
        # Create workbook and worksheet
        wb = Workbook()
        ws = wb.active
        ws.title = "Verification Results"
        
        # Sort results by paragraph number before writing to Excel
        def sort_by_paragraph_number(result):
            """Sort results by paragraph number for consistent Excel output"""
            # Try to get paragraph number from various possible column names
            par_number = result.get('Paragraph Number') or result.get('par_number') or result.get('Paragraph') or ''
            
            # Handle different paragraph number formats
            if isinstance(par_number, str):
                # Remove any non-alphanumeric characters and try to extract numbers
                import re
                numbers = re.findall(r'\d+', par_number)
                if numbers:
                    # Use the first number found for sorting
                    return (0, int(numbers[0]))  # Tuple: (0 = numeric, value)
                # If no numbers found, try to sort alphabetically
                return (1, par_number.lower())  # Tuple: (1 = alphabetical, value)
            elif isinstance(par_number, (int, float)):
                return (0, float(par_number))  # Tuple: (0 = numeric, value)
            elif par_number is None or par_number == '':
                # If paragraph number is None or empty, put at the end
                return (2, float('inf'))  # Tuple: (2 = invalid, infinity)
            else:
                # If paragraph number is invalid, put at the end
                return (2, float('inf'))  # Tuple: (2 = invalid, infinity)
        
        # Sort the results by paragraph number
        sorted_results = sorted(results, key=sort_by_paragraph_number)
        logger.info(f"Sorted {len(sorted_results)} results by paragraph number for Excel generation")
        
        # Define headers: fixed first two columns + dynamic output fields
        fixed_headers = ["Paragraph Number", "Statement"]
        dynamic_headers = []
        
        # Handle both dictionary and list formats for output_fields
        if output_fields:
            if isinstance(output_fields, dict):
                # If output_fields is a dictionary (e.g., {'par_number': True, 'is_accurate': True})
                # We need to load the field configuration to get names
                try:
                    output_fields_file = "config/output_fields_config.json"
                    if os.path.exists(output_fields_file):
                        with open(output_fields_file, 'r', encoding='utf-8') as f:
                            field_config = json.load(f)
                            
                        # Add headers for enabled fields
                        for field in field_config:
                            field_id = field.get('id')
                            if field_id in output_fields and output_fields[field_id] and not field.get('fixed', False):
                                field_name = field.get('name', field_id)
                                dynamic_headers.append(field_name)
                except Exception as e:
                    logger.error(f"Error loading field configuration: {e}")
                    # Fallback: use field IDs as headers
                    for field_id, enabled in output_fields.items():
                        if enabled and field_id not in ["par_number", "par_context"]:
                            dynamic_headers.append(field_id)
            else:
                # If output_fields is a list of field objects
                for field in output_fields:
                    if field.get("enabled", True) and field.get("id") not in ["par_number", "par_context"]:
                        dynamic_headers.append(field.get("name", field.get("id")))
        
        all_headers = fixed_headers + dynamic_headers
        ws.append(all_headers)
        
        # Add data rows
        for i, result in enumerate(sorted_results):
            try:
                # Ensure result is a dictionary
                if not isinstance(result, dict):
                    logger.error(f"Result {i} is not a dictionary: {type(result)} - {result}")
                    # Skip this result or create a default structure
                    continue
                
                # Fixed first two columns - try to get from result or use defaults
                par_number = result.get('Paragraph Number', result.get('par_number', 'N/A'))
                statement_content = result.get('Statement', result.get('par_context', 'N/A'))
                
                row_data = [par_number, statement_content]
                
                # Dynamic columns based on output fields
                if output_fields:
                    if isinstance(output_fields, dict):
                        # If output_fields is a dictionary, we need to get field names from config
                        try:
                            output_fields_file = "config/output_fields_config.json"
                            if os.path.exists(output_fields_file):
                                with open(output_fields_file, 'r', encoding='utf-8') as f:
                                    field_config = json.load(f)
                                    
                                # Add values for enabled fields
                                for field in field_config:
                                    field_id = field.get('id')
                                    if field_id in output_fields and output_fields[field_id] and not field.get('fixed', False):
                                        field_name = field.get('name', field_id)
                                        field_value = result.get(field_name, result.get(field_id, ""))
                                        row_data.append(field_value)
                        except Exception as e:
                            logger.error(f"Error loading field configuration for row generation: {e}")
                            # Fallback: use field IDs
                            for field_id, enabled in output_fields.items():
                                if enabled and field_id not in ["par_number", "par_context"]:
                                    field_value = result.get(field_id, "")
                                    row_data.append(field_value)
                    else:
                        # If output_fields is a list of field objects
                        for field in output_fields:
                            if field.get("enabled", True) and field.get("id") not in ["par_number", "par_context"]:
                                # Get the field value from the result using the field name
                                field_name = field.get("name", field.get("id"))
                                field_value = result.get(field_name, result.get(field.get("id"), ""))
                                row_data.append(field_value)
                
                ws.append(row_data)
                
            except Exception as e:
                logger.error(f"Error processing result {i}: {e}")
                # Add a row with error information
                error_row = [f"Error in row {i}", f"Error: {str(e)}"] + [""] * (len(all_headers) - 2)
                ws.append(error_row)
        
        # Convert to DataFrame for compatibility with existing code
        df = pd.DataFrame(sorted_results)
        
        # Try to save with a unique filename to avoid permission issues
        import time
        import os
        timestamp = int(time.time())
        output_file = f"verification_results_{timestamp}.xlsx"
        
        try:
            # First try to save to the standard filename
            try:
                # Try to remove existing file if it exists
                if os.path.exists("verification_results.xlsx"):
                    try:
                        os.remove("verification_results.xlsx")
                    except PermissionError:
                        logger.warning("Could not remove existing verification_results.xlsx (file may be open)")
                
                wb.save("verification_results.xlsx")
                output_file = "verification_results.xlsx"
                logger.info("Successfully saved to verification_results.xlsx")
            except PermissionError:
                logger.warning("Could not save to verification_results.xlsx (file may be open), using timestamped version")
                wb.save(output_file)
            except Exception as e:
                logger.error(f"Error saving to standard filename: {e}")
                wb.save(output_file)
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            # Try alternative location
            try:
                wb.save(f"temp_verification_results_{timestamp}.xlsx")
                output_file = f"temp_verification_results_{timestamp}.xlsx"
            except Exception as e2:
                logger.error(f"Failed to save results to any location: {e2}")
                raise e2
        
        update_status("completed", 100, "Verification completed", len(statements), len(statements),
                     log=f"Verification completed. Results saved to {output_file}")
        
        # Reset status to idle after 30 seconds to prevent continuous polling
        import asyncio
        async def reset_status_after_delay():
            await asyncio.sleep(30)
            update_status("idle", 0, "", 0, 0, "")
        
        # Start the reset task
        asyncio.create_task(reset_status_after_delay())
        
    except Exception as e:
        error_msg = f"Verification failed: {str(e)}"
        update_status("error", 0, "Verification failed", 0, 0, log=error_msg)
        logger.error(f"Verification error: {e}")
        
        # Ensure the status is properly set to error so the frontend can handle it
        from utils.status import update_status
        update_status("error", 0, "Error occurred", 0, 0, error_msg, error_msg)

@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": "1.0.0",
        "service": "VeriDoc AI"
    }

@app.get("/", response_class=HTMLResponse)
async def get_main_page(request: Request):
    """Serve the main HTML page with automatic mobile detection"""
    # Server-side mobile device detection
    user_agent = request.headers.get("user-agent", "").lower()
    
    # Check for mobile devices
    mobile_indicators = [
        "mobile", "android", "iphone", "ipad", "ipod", 
        "blackberry", "iemobile", "opera mini", "windows phone",
        "palm", "smartphone", "tablet", "kindle", "silk",
        "webos", "bada", "symbian", "meego"
    ]
    
    is_mobile = any(indicator in user_agent for indicator in mobile_indicators)
    
    # Redirect mobile users to mobile version
    if is_mobile:
        logger.info(f"Mobile device detected (User-Agent: {user_agent[:100]}...), redirecting to mobile version")
        return RedirectResponse(url="/mobile", status_code=302)
    
    logger.info(f"Desktop device detected (User-Agent: {user_agent[:100]}...), serving desktop version")
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeriDoc AI</title>
                      <link rel="icon" type="image/x-icon" href="/static/favicon.ico">
                  <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
                  <div class="header">
              <h1>VeriDoc AI</h1>
              <p>Automated Fact Verification and Document Consistency Platform</p>
              <div id="currentProjectDisplay" class="current-project-display">
                  <div class="project-dropdown-container">
                      <button id="currentProjectButton" class="project-name-button" onclick="toggleProjectDropdown()">
                          <span id="currentProjectName" class="project-name">Loading...</span>
                          <span class="dropdown-arrow">▼</span>
                      </button>
                      <div id="projectDropdown" class="project-dropdown hidden">
                          <!-- Project list will be populated here -->
                      </div>
                  </div>
              </div>
          </div>
        
        <div class="content">
                         <div class="tab-container">
                 <button class="tab active" data-tab="ask-me">Ask Me</button>
                 <button class="tab" data-tab="knowledge">Projects</button>
                 <button class="tab" data-tab="verify">Batch</button>
                 <button class="tab" data-tab="results">Results</button>
                 <button class="tab" data-tab="technical">Technical Info</button>
             </div>
            
            <div id="ask-me" class="tab-content active">
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
                </div>
            
            <div id="knowledge" class="tab-content">
                <h2>📚 Projects Management</h2>
                <p>Manage projects and their isolated knowledge databases</p>
                
                <div class="zones-container verification-zones">
                    <!-- Project Management Zone (Left Side - 50% width) -->
                    <div class="zone-left">
                        <h3>📁 Project Management</h3>
                        <p>Create and manage isolated knowledge databases for different projects.</p>
                        
                        <!-- Project List -->
                        <div class="project-list-container">
                            <div class="project-actions">
                                <button class="btn btn-primary" id="addProjectBtn">➕ Add Project</button>
                                <button class="btn" id="reorderProjectsBtn">🔄 Reorder</button>
                            </div>
                            
                            <div id="projectList" class="project-list">
                                <!-- Projects will be loaded here -->
                            </div>
                        </div>
                        
                        <!-- Project Management Modals -->
                        <div id="projectModal" class="modal hidden">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h3 id="projectModalTitle">Add New Project</h3>
                                    <span class="close" onclick="closeProjectModal()">&times;</span>
                                </div>
                                <div class="modal-body">
                                    <div class="input-group">
                                        <label for="projectName" class="form-label">Project Name:</label>
                                        <input type="text" id="projectName" placeholder="Enter project name" class="form-input">
                                    </div>
                                    <div class="input-group">
                                        <label for="projectDescription" class="form-label">Description:</label>
                                        <textarea id="projectDescription" placeholder="Enter project description (optional)" class="form-input" rows="3"></textarea>
                                    </div>
                                    <div class="button-group">
                                        <button class="btn btn-primary" id="saveProjectBtn">Save Project</button>
                                        <button class="btn" onclick="closeProjectModal()">Cancel</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <div id="deleteProjectModal" class="modal hidden">
                            <div class="modal-content">
                                <div class="modal-header">
                                    <h3>Delete Project</h3>
                                    <span class="close" onclick="closeDeleteProjectModal()">&times;</span>
                                </div>
                                <div class="modal-body">
                                    <p>Are you sure you want to delete the project "<span id="deleteProjectName"></span>"?</p>
                                    <p><strong>Warning:</strong> This will permanently delete the project and all its associated knowledge database files.</p>
                                    
                                    <div class="safe-word-container">
                                        <p>⚠️ Safety Confirmation Required</p>
                                        <p>To confirm deletion, type the word <strong>"delete"</strong> in the field below:</p>
                                        <input type="text" id="deleteConfirmationInput" placeholder="Type 'delete' to confirm" oninput="checkDeleteConfirmation()">
                                        <div id="deleteConfirmationStatus"></div>
                                    </div>
                                    
                                    <div class="button-group">
                                        <button class="btn danger" id="confirmDeleteProjectBtn" disabled>Delete Project</button>
                                        <button class="btn" onclick="closeDeleteProjectModal()">Cancel</button>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Knowledge Database Zone (Right Side - 50% width) -->
                    <div class="zone-right">
                        <h3>📚 Knowledge Database</h3>
                        <p id="currentProjectInfo">Select a project to manage its knowledge database.</p>
                        
                        <div id="knowledgeDatabaseContent" class="knowledge-database-content">
                            <!-- Upload Area -->
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
                                              <label for="bulkCategorySelect" class="form-label">Category:</label>
                                              <select id="bulkCategorySelect" class="form-select" title="Select a category to assign to the selected documents" aria-label="Select a category to assign to the selected documents">
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
                </div>
            </div>
            
            <div id="verify" class="tab-content">
                <h2>🔍 Batch Verification</h2>
                <p>Configure batch verification parameters and upload statements for verification against your knowledge database.</p>
                
                <div class="zones-container verification-zones">
                    <!-- Batch Verification Parameters Zone -->
                    <div class="zone-left">
                        <h3>⚙️ Batch Verification Parameters</h3>
                        <p>Configure the AI model parameters</p>
                        
                        
                        
                        <!-- Analysis Group -->
                        <div class="verification-group">
                            <div class="parameters-section">
                                <h5>🤖 Model Configuration</h5>
                                

                                
                                <div class="parameter-group">
                                    <label for="verificationRole">Role:</label>
                                    <select id="verificationRole" class="form-control">
                                        <option value="You are an experienced UK solicitor providing legal analysis." selected>UK Solicitor</option>
                                        <option value="You are a legal expert specializing in document verification.">Legal Expert</option>
                                        <option value="You are a fact-checking specialist.">Fact Checker</option>
                                        <option value="You are an AI assistant helping with document analysis.">AI Assistant</option>
                                    </select>
                                </div>
                                
                                                                 <div class="parameter-group">
                                     <label for="verificationCompletionTokens">Completion Tokens:</label>
                                     <input type="number" id="verificationCompletionTokens" class="form-control" value="4000" min="1000" max="8000" step="500">
                                 </div>
                                 
                                 <div class="parameter-group">
                                     <label for="verificationReasoningEffort">Reasoning Effort:</label>
                                     <select id="verificationReasoningEffort" class="form-control">
                                         <option value="low">Low</option>
                                         <option value="medium" selected>Medium</option>
                                         <option value="high">High</option>
                                     </select>
                                 </div>
                                 
                                 <div class="parameter-group">
                                     <label for="verificationVerbosity">Verbosity:</label>
                                     <select id="verificationVerbosity" class="form-control">
                                         <option value="low" selected>Low</option>
                                         <option value="medium">Medium</option>
                                         <option value="high">High</option>
                                     </select>
                                 </div>
                                

                            </div>
                            
                            <div class="parameters-section">
                                <h5>📋 Analysis Output Fields</h5>
                                
                                <div class="parameter-group">
                                    <div class="output-fields-container" id="verificationOutputFields">
                                        <!-- Dynamic verification output fields will be loaded here -->
                                    </div>
                                    <div class="field-actions">
                                        <button type="button" class="btn small secondary" onclick="addOutputField('verification')">➕ Add Field</button>
                                        <button type="button" class="btn small secondary" onclick="saveOutputFields()">💾 Save Fields</button>
                                        <button type="button" class="btn small secondary" onclick="resetVerificationFields()">🔄 Reset to Default</button>
                                    </div>
                                    <small>Configure which fields to include in analysis results. The "Question/instruction" is what the AI model will be asked to answer for each field.</small>
                                  </div>
                              </div>
                          </div>
                    </div>
                    
                    <!-- Statements for Verification Zone -->
                    <div class="zone-right">
                        <h3>📋 Statements for Verification</h3>
                        <p>Upload the Excel file containing statements/paragraphs that need to be verified against your knowledge database</p>
                        
                        <div class="statements-section">
                            <div class="upload-area statements" id="statementsUploadArea">
                                <p><strong>📊 Upload Statements File</strong></p>
                                <div class="column-requirements">
                                    <p><strong>📋 Column Requirements:</strong></p>
                                    <ul>
                                        <li><strong>Column 1:</strong> Paragraph number (required)</li>
                                        <li><strong>Column 2:</strong> Statement content (required)</li>
                                        <li><em>Additional columns will be ignored</em></li>
                                    </ul>
                                </div>
                                <p><small>Supports: Excel files (.xlsx, .xls) with at least 2 columns</small></p>
                                <input type="file" id="statementsFileInput" accept=".xlsx,.xls" class="file-input-hidden" aria-label="Upload statements file for verification" title="Upload statements file for verification">
                            </div>
                            
                            <!-- Button Container -->
                            <div class="flex-center">
                                <button id="startBtn" class="btn success large" disabled>Start Analysis</button>
                            </div>
                            
                            <!-- Processing Note (hidden by default) -->
                            <div id="processingNote" class="processing-note processing-note-hidden">
                                Quick analysis may take several minutes, please keep calm and wait for the analysis results
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
                 <div class="results-info">
                     <div class="current-file-info" id="currentFileInfo">
                         <span class="file-label">📁 Current File:</span>
                         <span class="file-name" id="currentFileName">No results loaded</span>

                     </div>
                     <div class="results-actions">
                         <button class="btn" id="loadResultsBtn">📊 Load Results</button>
                         <button class="btn success" id="downloadResultsBtn">💾 Download Results</button>
                         <button class="btn secondary" id="refreshResultsBtn" onclick="refreshResultsTable()">🔄 Refresh Table</button>
                         <button class="btn info" id="showFilesBtn" onclick="showResultsFiles()">📋 Show Files</button>
                     </div>
                 </div>
                 <div id="resultsTable"></div>
                 <div id="filesInfo" class="files-info">
                     <h3>📋 Available Results Files</h3>
                     <div id="filesList"></div>
                 </div>
             </div>
             
             <div id="technical" class="tab-content">
                 <h2>🔧 Technical Configuration</h2>
                 <p>Current technical parameters used for document quick analysis processing.</p>
                 
                 <div class="tech-info-grid">
                     <div class="tech-card">
                         <h3>🤖 AI Models</h3>
                         <div class="tech-item">
                             <span class="tech-label">Judge Model:</span>
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
         
         // Project management variables
         let projects = {};
         let currentProject = null;
         let editingProject = null;
        
                 // Initialize when DOM is loaded
         document.addEventListener('DOMContentLoaded', function() {
            // Add click-outside handler for project dropdown
            document.addEventListener('click', function(event) {
                const dropdown = document.getElementById('projectDropdown');
                const button = document.getElementById('currentProjectButton');
                
                if (dropdown && button && !dropdown.contains(event.target) && !button.contains(event.target)) {
                    closeProjectDropdown();
                }
            });
            
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
             
             // Load projects and set up project management
             loadProjects();
             
             initializeEventListeners();
             initializeOutputFields(); // Initialize dynamic output fields
             loadCaches(); // Auto-load caches on startup
             
             // Re-initialize event listeners after a short delay to ensure all elements are loaded
             setTimeout(() => {
                 initializeEventListeners();
                 ensureAllButtonsWorking();
                 retryButtonInitialization();
             }, 500);
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
            

            
            // Initialize dynamic output fields
            initializeOutputFields();
            
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
                manageCategoriesBtn.addEventListener('click', () => {
                    openCategoryModal();
                });
            }
            
            const bulkAssignBtn = document.getElementById('bulkAssignBtn');
            if (bulkAssignBtn) {
                bulkAssignBtn.addEventListener('click', () => {
                    openBulkModal();
                });
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
             
             const refreshResultsBtn = document.getElementById('refreshResultsBtn');
             if (refreshResultsBtn) {
                 refreshResultsBtn.addEventListener('click', refreshResultsTable);
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
             
             // Initialize analysis parameters
             // (No longer needed since we removed the analysis type selector)
             
             // Project management event listeners
                         const addProjectBtn = document.getElementById('addProjectBtn');
            if (addProjectBtn) {
                addProjectBtn.addEventListener('click', () => {
                    openProjectModal();
                });
            }
             
             const saveProjectBtn = document.getElementById('saveProjectBtn');
             if (saveProjectBtn) {
                 saveProjectBtn.addEventListener('click', saveProject);
             }
             
             const confirmDeleteProjectBtn = document.getElementById('confirmDeleteProjectBtn');
             if (confirmDeleteProjectBtn) {
                 confirmDeleteProjectBtn.addEventListener('click', confirmDeleteProject);
             }
             
             const reorderProjectsBtn = document.getElementById('reorderProjectsBtn');
             if (reorderProjectsBtn) {
                 reorderProjectsBtn.addEventListener('click', toggleProjectReorderMode);
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
            
            // If switching to knowledge tab, ensure buttons are initialized
            if (tabName === 'knowledge') {
                setTimeout(() => {
                    retryButtonInitialization();
                }, 100);
            }
            
            // Auto-refresh results when switching to Results tab
            if (tabName === 'results') {
                loadResults();
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
                const indexFilesBtn = document.getElementById('indexFilesBtn');
                
                if (startBtn) {
                    const hasKnowledge = knowledgeFiles.length > 0 || selectedCaches.length > 0;
                    const hasStatements = statementsFile !== null;
                    startBtn.disabled = !(hasKnowledge && hasStatements);
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
                

                
                if (!askMeInput || !askMeBtn || !conversationMessages) {
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
                    
                    if (response.ok) {
                        const data = await response.json();
                        
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
                        
                        // Remove loading animation
                        removeLoadingMessage(loadingElement);
                        
                        addMessageToConversation('assistant', `Error: ${errorData.detail}`, timestamp);
                    }
                } catch (error) {
                    
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
                    alert('No conversation to copy');
                    return;
                }
                
                let conversationText = '';
                
                messages.forEach(message => {
                    const role = message.classList.contains('user') ? 'User' : 'Assistant';
                    const time = message.querySelector('.message-time')?.textContent || '';
                    const content = message.querySelector('.message-content')?.textContent || '';
                    
                    conversationText += '[' + time + '] ' + role + ':\\n' + content + '\\n\\n';
                });
                
                // Remove trailing newlines
                conversationText = conversationText.trim();
                
                const copyBtn = document.querySelector('.conversation-copy-btn');
                
                // Try modern clipboard API first (works on HTTPS/localhost)
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(conversationText).then(() => {
                        // Visual feedback
                        copyBtn.classList.add('copied');
                        copyBtn.textContent = '✓';
                        
                        setTimeout(() => {
                            copyBtn.classList.remove('copied');
                            copyBtn.textContent = '📋';
                        }, 2000);
                    }).catch(err => {
                        copyWithFallback(conversationText, copyBtn);
                    });
                } else {
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
                    textArea.style.opacity = '0';
                    document.body.appendChild(textArea);
                    textArea.focus();
                    textArea.select();
                    
                    const successful = document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    if (successful) {
                        // Visual feedback
                        copyBtn.classList.add('copied');
                        copyBtn.textContent = '✓';
                        
                        setTimeout(() => {
                            copyBtn.classList.remove('copied');
                            copyBtn.textContent = '📋';
                        }, 2000);
                    } else {
                        // Show text in alert as last resort
                        alert('Copy failed. Here is the conversation text:\\n\\n' + text.substring(0, 500) + (text.length > 500 ? '...' : ''));
                    }
                } catch (err) {
                    // Show text in alert as last resort
                    alert('Copy failed. Here is the conversation text:\\n\\n' + text.substring(0, 500) + (text.length > 500 ? '...' : ''));
                }
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

                    } catch (error) {
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
                        
                        const formData = new FormData();
                        files.forEach(file => {
                            formData.append('files', file);
                        });
                        
                        // Add conversation ID if available
                        if (currentConversationId) {
                            formData.append('conversation_id', currentConversationId);
                        }
                        
                        try {
                            const response = await fetch('/api/ask-me-upload', {
                                method: 'POST',
                                body: formData
                            });
                            
                            if (response.ok) {
                                const data = await response.json();
                                askMeFiles = data.uploaded_files;
                                
                                // Store conversation ID if provided
                                if (data.conversation_id) {
                                    currentConversationId = data.conversation_id;
                                }
                                
                                displayAskMeFiles();
                                updateAdditionalContext();
                            } else {
                                const errorData = await response.json();
                                alert('Upload failed: ' + (errorData.detail || 'Unknown error'));
                            }
                        } catch (error) {
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
                    } catch (error) {
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
                    } catch (error) {
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
             const result = await uploadFile(file, 'statements');
             // Load and display statements after upload using the sanitized filename
             if (result && result.filename) {
                 // Validate the Excel file structure after upload
                 await validateExcelFile(result.filename);
                 await loadStatementsData(result.filename);
             } else {
                 await loadStatementsData(file.name);
             }
         }
        
        async function validateExcelFile(filename) {
            try {
                const response = await fetch('/api/validate-excel', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        statements_file: filename
                    })
                });
                
                const result = await response.json();
                
                if (result.valid) {
                    // Show success message with file info
                    const successMsg = `✅ ${result.message}\n\nFile: ${result.file_info.filename}\nColumns: ${result.file_info.columns_used}\nStatements: ${result.statement_count}`;
                    showNotification(successMsg, 'success');
                    
                    // Update the file info display with validation results
                    updateStatementsFileInfoWithValidation(result);
                } else {
                    // Show error message with details
                    const errorMsg = `❌ ${result.message}\n\nRequirements:\n• Column 1: Paragraph number (required)\n• Column 2: Statement content (required)\n• Additional columns will be ignored`;
                    showNotification(errorMsg, 'error');
                    
                    // Optionally, you could disable the start button or remove the file
                    // For now, just show the error
                }
                
            } catch (error) {
                showNotification('❌ Failed to validate Excel file. Please check the file format.', 'error');
            }
        }
        
        function showNotification(message, type = 'info') {
            // Create notification element
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 15px 20px;
                border-radius: 8px;
                color: white;
                font-weight: 500;
                z-index: 10000;
                max-width: 400px;
                word-wrap: break-word;
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                animation: slideIn 0.3s ease-out;
            `;
            
            // Set background color based on type
            switch (type) {
                case 'success':
                    notification.style.backgroundColor = '#28a745';
                    break;
                case 'error':
                    notification.style.backgroundColor = '#dc3545';
                    break;
                case 'warning':
                    notification.style.backgroundColor = '#ffc107';
                    notification.style.color = '#212529';
                    break;
                default:
                    notification.style.backgroundColor = '#17a2b8';
            }
            
            // Set message content
            notification.textContent = message;
            
            // Add to page
            document.body.appendChild(notification);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 5000);
            
            // Add click to dismiss
            notification.addEventListener('click', () => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            });
        }
        
        function updateStatementsFileInfoWithValidation(validationResult) {
            const fileInfo = document.getElementById('statementsFileInfo');
            if (!fileInfo || !statementsFile) return;
            
            // Update the file info display with validation results
            const fileItem = fileInfo.querySelector('.file-item');
            if (fileItem) {
                // Add validation status
                const validationStatus = document.createElement('div');
                validationStatus.className = 'validation-status';
                validationStatus.style.cssText = `
                    margin-top: 8px;
                    padding: 8px;
                    background: #d4edda;
                    border: 1px solid #c3e6cb;
                    border-radius: 4px;
                    font-size: 12px;
                    color: #155724;
                `;
                validationStatus.innerHTML = `
                    <strong>✓ Validated:</strong> ${validationResult.statement_count} statements found<br>
                    <strong>Columns:</strong> ${validationResult.file_info.columns_used}<br>
                    <strong>File:</strong> ${validationResult.file_info.filename}
                `;
                
                fileItem.appendChild(validationStatus);
            }
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
                
                return result;
                
            } catch (error) {
                return null;
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
                // Get verification parameters
                const verificationParams = {
                    role: document.getElementById('verificationRole').value,
                    completion_tokens: parseInt(document.getElementById('verificationCompletionTokens').value),
                    reasoning_effort: document.getElementById('verificationReasoningEffort').value,
                    verbosity: document.getElementById('verificationVerbosity').value
                };
                
                // Get output field selections and descriptions
                const outputFields = getOutputFields('verification');
                const outputFieldDescriptions = getOutputFieldDescriptions('verification');
                
                // Use the standard processing endpoint
                const endpoint = '/api/process';
                
                const response = await fetch(endpoint, {
                     method: 'POST',
                     headers: {
                         'Content-Type': 'application/json; charset=utf-8'
                     },
                     body: JSON.stringify({
                         selected_caches: selectedCaches,
                         statements_file: statementsFile.name,
                         knowledge_files: knowledgeFiles.map(f => f.name),
                        selected_statements: selectedStatements,
                        verification_params: verificationParams,
                        output_fields: outputFields,
                        output_field_descriptions: outputFieldDescriptions
                     })
                 });
                 
                 if (!response.ok) {
                     throw new Error('Failed to start processing');
                 }
                 
                 // Start status polling
                 startStatusPolling();
                 
             } catch (error) {
                 alert('Failed to start processing');
                 
                 // Reset button and hide notes on error
                 startBtn.disabled = false;
                 startBtn.className = 'btn success large';
                startBtn.innerHTML = 'Start Analysis';
                 processingNote.style.display = 'none';
                 completionNote.style.display = 'none';
             }
         }
        
        function startStatusPolling() {
            // Clear any existing polling
            if (statusPolling) {
                clearInterval(statusPolling);
            }
            
            // Add a timeout to prevent infinite polling (5 minutes max)
            const maxPollingTime = 5 * 60 * 1000; // 5 minutes
            const startTime = Date.now();
            
            statusPolling = setInterval(async () => {
                // Check if we've been polling too long
                if (Date.now() - startTime > maxPollingTime) {
                    stopStatusPolling();
                    return;
                }
                try {
                    // Check processing status
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
                            startBtn.innerHTML = '✅ Analysis Completed';
                            startBtn.disabled = true;
                            
                            // Hide processing note and show completion note
                            processingNote.style.display = 'none';
                            completionNote.style.display = 'block';
                        } else {
                            // Reset on error
                            startBtn.disabled = false;
                            startBtn.className = 'btn success large';
                            startBtn.innerHTML = 'Start Analysis';
                            processingNote.style.display = 'none';
                            completionNote.style.display = 'none';
                        }
                    }
                } catch (error) {
                    // Stop polling on error to prevent infinite requests
                    stopStatusPolling();
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
                // First get the current file info
                const filesResponse = await fetch('/api/results/files');
                const filesData = await filesResponse.json();
                
                if (filesData.latest_file) {
                    currentResultsFile = filesData.latest_file;
                    document.getElementById('currentFileName').textContent = currentResultsFile;
                    

                } else {
                    currentResultsFile = null;
                    document.getElementById('currentFileName').textContent = 'No results file found';
                    

                }
                
                // Then load the results
                const response = await fetch('/api/results');
                const results = await response.json();
                await displayResults(results);
            } catch (error) {
                alert('Failed to load results');
                currentResultsFile = null;
                document.getElementById('currentFileName').textContent = 'Error loading results';
                

            }
        }
        
        async function refreshResultsTable() {
            try {
                await loadResults();
            } catch (error) {
            }
        }
        
        async function showResultsFiles() {
            try {
                const response = await fetch('/api/results/files');
                const data = await response.json();
                
                const filesInfo = document.getElementById('filesInfo');
                const filesList = document.getElementById('filesList');
                
                if (data.files && data.files.length > 0) {
                    let filesHTML = '<div class="files-grid">';
                    
                    data.files.forEach(file => {
                        const isCurrent = file.filename === currentResultsFile;
                        const currentClass = isCurrent ? 'current-file' : '';
                        
                        filesHTML += `
                            <div class="file-item ${currentClass}" onclick="selectResultsFile('${file.filename}')">
                                <div class="file-header">
                                    <span class="file-type ${file.type}">${file.type}</span>
                                    ${isCurrent ? '<span class="current-badge">Current</span>' : ''}
                                </div>
                                <div class="file-name">${file.filename}</div>
                                <div class="file-details">
                                    <div class="file-info-row">
                                        <span class="file-info-label">Size:</span>
                                        <span class="file-info-value">${file.size_mb} MB</span>
                                    </div>
                                    <div class="file-info-row">
                                        <span class="file-info-label">Modified:</span>
                                        <span class="file-info-value">${file.modified_readable}</span>
                                    </div>
                                </div>
                            </div>
                        `;
                    });
                    
                    filesHTML += '</div>';
                    filesList.innerHTML = filesHTML;
                } else {
                    filesList.innerHTML = '<p>No results files found</p>';
                }
                
                filesInfo.style.display = 'block';
                
            } catch (error) {
                alert('Failed to load files information');
            }
        }
        
        let currentResultsFile = null;
        
        async function selectResultsFile(filename) {
            try {
                // Update current file selection
                currentResultsFile = filename;
                
                // Update the current file display
                const currentFileName = document.getElementById('currentFileName');
                if (currentFileName) {
                    currentFileName.textContent = filename;
                }
                

                
                // Load results from the selected file
                await loadResultsFromFile(filename);
                
                // Update the files display to show the new selection
                await showResultsFiles();
                
                // Show success message
                const successMsg = document.createElement('div');
                successMsg.className = 'success-message';
                successMsg.textContent = `Switched to file: ${filename}`;
                successMsg.style.cssText = 'background: #d4edda; color: #155724; padding: 10px; border-radius: 4px; margin: 10px 0; border: 1px solid #c3e6cb;';
                
                const resultsContainer = document.getElementById('resultsContainer');
                if (resultsContainer) {
                    resultsContainer.insertBefore(successMsg, resultsContainer.firstChild);
                    
                    // Remove the message after 3 seconds
                    setTimeout(() => {
                        if (successMsg.parentNode) {
                            successMsg.remove();
                        }
                    }, 3000);
                }
                
            } catch (error) {
                alert(`Failed to select file: ${filename}`);
            }
        }
        
        async function loadResultsFromFile(filename) {
            try {
                const response = await fetch(`/api/results?filename=${encodeURIComponent(filename)}`);
                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }
                
                const results = await response.json();
                
                // Display the results
                await displayResults(results);
                
            } catch (error) {
                throw error;
            }
        }
        

        

        
        async function displayResults(results) {
            const resultsTable = document.getElementById('resultsTable');
            
            if (!results || results.length === 0) {
                resultsTable.innerHTML = '<p>No results found</p>';
                return;
            }
            

            
            // Load current output fields configuration to determine table structure
            let outputFields = [];
            try {
                const response = await fetch('/api/load-output-fields');
                const data = await response.json();
                outputFields = data.fields || [];
            } catch (error) {
                // Fallback to default fields if API fails
                outputFields = defaultVerificationFields;
            }
            
            // Build table headers dynamically
            let tableHTML = '<table class="results-table"><thead><tr>';
            
            // Fixed columns - always present
            tableHTML += '<th>Paragraph Number</th>';
            tableHTML += '<th class="par-content">Statement</th>';
            
            // Dynamic columns based on output fields
            const dynamicFields = outputFields.filter(field => 
                field.enabled && !field.fixed && field.id !== 'par_number' && field.id !== 'par_context'
            );
            
            dynamicFields.forEach(field => {
                tableHTML += `<th class="center-align">${field.name}</th>`;
            });
            
            tableHTML += '</tr></thead><tbody>';
            
            // Build table rows
            results.forEach((result, index) => {
                tableHTML += '<tr>';
                
                // Fixed columns
                const parNumber = result['Paragraph Number'] || result['Par Number'] || '';
                tableHTML += `<td>${parNumber}</td>`;
                
                const statementContent = result['Statement'] || result['Par Context'] || '';
                const shortContent = statementContent.length > 200 ? 
                    statementContent.substring(0, 200) + '...' : statementContent;
                
                tableHTML += `<td class="par-content" onclick="toggleParContent(${index})">` +
                    `<div class="par-content-text" id="par-content-${index}" data-full-content="${statementContent.replace(/"/g, '&quot;')}">${shortContent}</div>` +
                    '</td>';
                
                // Dynamic columns
                dynamicFields.forEach(field => {
                    const fieldValue = result[field.name] || result[field.id] || '';
                    tableHTML += `<td class="center-align">${fieldValue}</td>`;
                });
                
                tableHTML += '</tr>';
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
                alert('Failed to download results');
            }
        }
        
        
        
        function updateAnalysisParameters() {
            // This function is kept for compatibility but no longer needed
            // since we removed the analysis type selector
        }
        
        // Default output field configurations
        const defaultVerificationFields = [
            { id: 'par_number', name: 'Paragraph Number from the input file', description: '', enabled: true, fixed: true },
            { id: 'par_context', name: 'Statement content from the input file', description: '', enabled: true, fixed: true },
            { id: 'is_accurate', name: 'Document Reference', description: 'List documents which are mentioned in the statement', enabled: true, fixed: false },
            { id: 'field_1756803554789', name: 'Fact/Finding', description: 'Check the statement below if there are Facts or Findings in respect of Primecap only in this particular statement? (Answer short Fact/Finding))', enabled: true, fixed: false },
            { id: 'field_1756803586927', name: 'Findings', description: 'List the findings in respect of Primecap in (a),(b),(c) list format', enabled: true, fixed: false }
        ];
        
        // Global variables to store current field configurations
        let verificationFields = JSON.parse(JSON.stringify(defaultVerificationFields));
        
        function initializeOutputFields() {
            loadSavedOutputFields();
        }
        
        function loadOutputFields(type, fields) {
            const container = document.getElementById('verificationOutputFields');
            
            if (container) {
                container.innerHTML = '';
                
                fields.forEach(field => {
                    addOutputFieldItem(container, field, type);
                });
            } else {
                console.error(`Container not found for type: ${type}`);
            }
        }
        
        function addOutputField(type) {
            const container = document.getElementById('verificationOutputFields');
            const newField = {
                id: 'field_' + Date.now(),
                name: 'New Field',
                description: 'Description for the AI model',
                enabled: true
            };
            
            addOutputFieldItem(container, newField, type);
            verificationFields.push(newField);
        }
        
        function addOutputFieldItem(container, field, type) {
            const fieldDiv = document.createElement('div');
            fieldDiv.className = `output-field-item ${field.fixed ? 'fixed-field' : ''}`;
            
            // For fixed fields, disable editing and removal
            const isFixed = field.fixed || field.id === 'par_number' || field.id === 'par_context';
            
            fieldDiv.innerHTML = `
                <div class="field-header">
                    <input type="checkbox" ${field.enabled ? 'checked' : ''} onchange="updateFieldEnabled(this, '${field.id}', '${type}')" ${isFixed ? 'disabled' : ''}>
                    <div class="field-name">
                        <input type="text" value="${field.name}" placeholder="Field Name" onchange="updateFieldName(this, '${field.id}', '${type}')" ${isFixed ? 'readonly' : ''}>
                    </div>
                    <div class="field-actions">
                        ${isFixed ? '' : `<button type="button" class="btn-remove" onclick="removeOutputField('${field.id}', '${type}')">🗑️</button>`}
                    </div>
                </div>
                ${isFixed ? '' : `
                <div class="field-description">
                    <textarea placeholder="Question/instruction for AI model" onchange="updateFieldDescription(this, '${field.id}', '${type}')">${field.description}</textarea>
                </div>
                `}
            `;
            container.appendChild(fieldDiv);
        }
        
        function updateFieldEnabled(checkbox, fieldId, type) {
            const field = verificationFields.find(f => f.id === fieldId);
            if (field) {
                field.enabled = checkbox.checked;
            }
        }
        
        function updateFieldName(input, fieldId, type) {
            const field = verificationFields.find(f => f.id === fieldId);
            if (field && (field.fixed || fieldId === 'par_number' || fieldId === 'par_context')) {
                alert('Cannot rename fixed columns. The first two columns (Paragraph Number and Statement) cannot be modified.');
                input.value = field.name; // Reset to original value
                return;
            }
            if (field) {
                field.name = input.value;
            }
        }
        
        function updateFieldDescription(textarea, fieldId, type) {
            const field = verificationFields.find(f => f.id === fieldId);
            if (field && (field.fixed || fieldId === 'par_number' || fieldId === 'par_context')) {
                alert('Cannot modify fixed columns. The first two columns (Paragraph Number and Statement) cannot be modified.');
                textarea.value = field.description; // Reset to original value
                return;
            }
            if (field) {
                field.description = textarea.value;
            }
        }
        
        function removeOutputField(fieldId, type) {
            const field = verificationFields.find(f => f.id === fieldId);
            if (field && (field.fixed || fieldId === 'par_number' || fieldId === 'par_context')) {
                alert('Cannot remove fixed columns. The first two columns (Paragraph Number and Statement) are required and cannot be deleted.');
                return;
            }
            
            const index = verificationFields.findIndex(f => f.id === fieldId);
            if (index > -1) {
                verificationFields.splice(index, 1);
                loadOutputFields(type, verificationFields);
            }
        }
        
        function resetVerificationFields() {
            verificationFields = JSON.parse(JSON.stringify(defaultVerificationFields));
            loadOutputFields('verification', verificationFields);
        }
        
        function getOutputFields(type) {
            const result = {};
            verificationFields.forEach(field => {
                if (field.enabled) {
                    result[field.id] = true;
                }
            });
            return result;
        }
        
        function getOutputFieldDescriptions(type) {
            const result = {};
            verificationFields.forEach(field => {
                if (field.enabled) {
                    result[field.id] = field.description;
                }
            });
            return result;
        }
        
        function saveOutputFields() {
            try {
                // Save to localStorage
                localStorage.setItem('verificationOutputFields', JSON.stringify(verificationFields));
                
                // Save to server
                fetch('/api/save-output-fields', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8'
                    },
                    body: JSON.stringify({
                        fields: verificationFields
                    })
                }).then(response => {
                    if (response.ok) {
                        // Refresh the results table to reflect new field structure
                        loadResults();
                    } else {
                        console.error('Failed to save fields to server');
                    }
                }).catch(error => {
                    console.error('Error saving fields to server:', error);
                });
                
                // Show success message
                const saveBtn = document.querySelector('button[onclick="saveOutputFields()"]');
                const originalText = saveBtn.textContent;
                saveBtn.textContent = '✅ Saved!';
                saveBtn.disabled = true;
                
                setTimeout(() => {
                    saveBtn.textContent = originalText;
                    saveBtn.disabled = false;
                }, 2000);
                
            } catch (error) {
                console.error('Error saving output fields:', error);
                alert('Failed to save output fields');
            }
        }
        
        function loadSavedOutputFields() {
            try {
                // First try to load from server
                fetch('/api/load-output-fields')
                    .then(response => response.json())
                    .then(data => {
                        if (data.fields && data.fields.length > 0) {
                            // Check if the loaded fields contain old descriptions that need to be cleared
                            // Only check for very specific old patterns that are definitely outdated (without fixed column suffix)
                            const hasOldDescriptions = data.fields.some(field => 
                                field.description && (
                                    field.description === 'Paragraph number from the statement' ||
                                    field.description === 'Statement content from the input file'
                                )
                            );
                            
                            if (hasOldDescriptions) {
                                console.log('Detected old descriptions, clearing storage and using new defaults');
                                // Only clear if we haven't already cleared recently
                                const lastClearTime = localStorage.getItem('verificationFieldsLastCleared');
                                const now = Date.now();
                                if (!lastClearTime || (now - parseInt(lastClearTime)) > 60000) { // 1 minute cooldown
                                clearOutputFieldsStorage();
                                    localStorage.setItem('verificationFieldsLastCleared', now.toString());
                                } else {
                                    console.log('Skipping clear due to recent clear operation');
                                    verificationFields = data.fields;
                                    loadOutputFields('verification', verificationFields);
                                }
                            } else {
                                verificationFields = data.fields;
                                loadOutputFields('verification', verificationFields);
                            }
                        } else {
                            // Fall back to localStorage
                            loadFromLocalStorage();
                        }
                    })
                    .catch(error => {
                        console.error('Error loading from server:', error);
                        // Fall back to localStorage
                        loadFromLocalStorage();
                    });
            } catch (error) {
                console.error('Error loading saved output fields:', error);
                // Fall back to defaults
                verificationFields = JSON.parse(JSON.stringify(defaultVerificationFields));
                loadOutputFields('verification', verificationFields);
            }
        }
        
        function loadFromLocalStorage() {
            try {
                const saved = localStorage.getItem('verificationOutputFields');
                if (saved) {
                    const parsedFields = JSON.parse(saved);
                    
                    // Check if the saved fields contain old descriptions that need to be cleared
                    // Only check for very specific old patterns that are definitely outdated (without fixed column suffix)
                    const hasOldDescriptions = parsedFields.some(field => 
                        field.description && (
                            field.description === 'Paragraph number from the statement' ||
                            field.description === 'Statement content from the input file'
                        )
                    );
                    
                    if (hasOldDescriptions) {
                        console.log('Detected old descriptions in localStorage, clearing and using new defaults');
                        // Only clear if we haven't already cleared recently
                        const lastClearTime = localStorage.getItem('verificationFieldsLastCleared');
                        const now = Date.now();
                        if (!lastClearTime || (now - parseInt(lastClearTime)) > 60000) { // 1 minute cooldown
                        clearOutputFieldsStorage();
                            localStorage.setItem('verificationFieldsLastCleared', now.toString());
                        } else {
                            console.log('Skipping clear due to recent clear operation');
                            verificationFields = parsedFields;
                            loadOutputFields('verification', verificationFields);
                        }
                    } else {
                        verificationFields = parsedFields;
                        loadOutputFields('verification', verificationFields);
                    }
                } else {
                    // Use defaults
                    verificationFields = JSON.parse(JSON.stringify(defaultVerificationFields));
                    loadOutputFields('verification', verificationFields);
                }
            } catch (error) {
                console.error('Error loading from localStorage:', error);
                // Use defaults
                verificationFields = JSON.parse(JSON.stringify(defaultVerificationFields));
                loadOutputFields('verification', verificationFields);
            }
        }
        
        function clearOutputFieldsStorage() {
            console.log('Clearing output fields storage due to old descriptions detected');
            localStorage.removeItem('verificationOutputFields');
            // Don't remove the cooldown timer - keep it to prevent repeated resets
            verificationFields = JSON.parse(JSON.stringify(defaultVerificationFields));
            loadOutputFields('verification', verificationFields);
            console.log('Output fields storage cleared and reset to defaults');
            
            // Show user notification
            showNotification('Output fields have been reset to defaults due to outdated configuration', 'info');
        }
        
        function manualResetOutputFields() {
            if (confirm('Are you sure you want to reset all output fields to defaults? This will clear your current configuration.')) {
                clearOutputFieldsStorage();
                showNotification('Output fields have been manually reset to defaults', 'success');
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
                
                // Sync localStorage categories to server if they exist
                const savedCategories = localStorage.getItem('veridoc_cache_categories');
                if (savedCategories) {
                    const localCategories = JSON.parse(savedCategories);
                    if (Object.keys(localCategories).length > 0) {
                        try {
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

                  
                  // Get categories from localStorage
                  const savedCategories = localStorage.getItem('veridoc_cache_categories');
                  if (!savedCategories) {
                      alert('No categories found in localStorage to sync.');
                      return;
                  }
                  
                  const localCategories = JSON.parse(savedCategories);
                  
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
            console.log('openCategoryModal called');
            const modal = document.getElementById('categoryModal');
            console.log('Category modal found:', !!modal);
            if (!modal) {
                console.error('Category modal not found!');
                return;
            }
            console.log('Removing hidden class from category modal');
            modal.classList.remove('hidden');
            console.log('Modal classes after removal:', modal.className);
            console.log('Modal display style:', window.getComputedStyle(modal).display);
            
            // Fallback: Force display if CSS doesn't work
            if (window.getComputedStyle(modal).display === 'none') {
                console.log('CSS not working, forcing display for category modal');
                modal.style.display = 'block';
            }
            
            await loadCategoryList();
        }
        
        function closeCategoryModal() {
            const modal = document.getElementById('categoryModal');
            modal.classList.add('hidden');
            // Reset inline style if it was set
            modal.style.display = '';
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
            console.log('openBulkModal called');
            const modal = document.getElementById('bulkModal');
            console.log('Bulk modal found:', !!modal);
            if (!modal) {
                console.error('Bulk modal not found!');
                return;
            }
            console.log('Removing hidden class from bulk modal');
            modal.classList.remove('hidden');
            console.log('Modal classes after removal:', modal.className);
            console.log('Modal display style:', window.getComputedStyle(modal).display);
            
            // Fallback: Force display if CSS doesn't work
            if (window.getComputedStyle(modal).display === 'none') {
                console.log('CSS not working, forcing display for bulk modal');
                modal.style.display = 'block';
            }
            
            await loadBulkCategoryOptions();
            updateBulkSelectionInfo();
        }
        
        function closeBulkModal() {
            const modal = document.getElementById('bulkModal');
            modal.classList.add('hidden');
            // Reset inline style if it was set
            modal.style.display = '';
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
            const projectModal = document.getElementById('projectModal');
            const deleteProjectModal = document.getElementById('deleteProjectModal');
            
            if (event.target === categoryModal) {
                closeCategoryModal();
            }
            if (event.target === bulkModal) {
                closeBulkModal();
            }
            if (event.target === projectModal) {
                closeProjectModal();
            }
            if (event.target === deleteProjectModal) {
                closeDeleteProjectModal();
            }
        }
        
        // ========================================
        // PROJECT MANAGEMENT FUNCTIONS
        // ========================================
        
        async function loadProjects() {
            try {
                const response = await fetch('/api/projects');
                const data = await response.json();
                
                if (response.ok) {
                    projects = data.projects;
                    currentProject = data.current_project;
                    
                    // Update header display
                    updateCurrentProjectDisplay();
                    
                    // Load project list
                    loadProjectList();
                    
                    // Update knowledge database info
                    updateKnowledgeDatabaseInfo();
                } else {
                    console.error('Error loading projects:', data.detail);
                }
            } catch (error) {
                console.error('Error loading projects:', error);
            }
        }
        
        function updateCurrentProjectDisplay() {
            const projectNameElement = document.getElementById('currentProjectName');
            if (projectNameElement) {
                projectNameElement.textContent = currentProject || 'No Project Selected';
            }
            
            // Update project dropdown
            updateProjectDropdown();
        }
        
        function updateProjectDropdown() {
            const dropdown = document.getElementById('projectDropdown');
            if (!dropdown) return;
            
            dropdown.innerHTML = '';
            
            Object.keys(projects).forEach(projectName => {
                const projectItem = document.createElement('div');
                projectItem.className = 'project-dropdown-item';
                if (projectName === currentProject) {
                    projectItem.classList.add('active');
                }
                projectItem.textContent = projectName;
                projectItem.onclick = () => {
                    selectProject(projectName);
                    closeProjectDropdown();
                };
                dropdown.appendChild(projectItem);
            });
        }
        
        function toggleProjectDropdown() {
            const dropdown = document.getElementById('projectDropdown');
            if (!dropdown) return;
            
            dropdown.classList.toggle('hidden');
            
            // Update dropdown arrow direction
            const arrow = document.querySelector('.dropdown-arrow');
            if (dropdown.classList.contains('hidden')) {
                arrow.textContent = '▼';
            } else {
                arrow.textContent = '▲';
            }
        }
        
        function closeProjectDropdown() {
            const dropdown = document.getElementById('projectDropdown');
            if (!dropdown) return;
            
            dropdown.classList.add('hidden');
            
            // Update dropdown arrow direction
            const arrow = document.querySelector('.dropdown-arrow');
            arrow.textContent = '▼';
        }
        
        function loadProjectList() {
            const projectList = document.getElementById('projectList');
            if (!projectList) return;
            
            if (Object.keys(projects).length === 0) {
                projectList.innerHTML = `
                    <div class="project-list-empty">
                        <div class="project-list-empty-icon">📁</div>
                        <div class="project-list-empty-text">No projects found</div>
                        <button class="btn btn-primary" onclick="openProjectModal()">Create Your First Project</button>
                    </div>
                `;
                return;
            }
            
            const projectItems = Object.values(projects).map(project => {
                const isSelected = project.name === currentProject;
                const createdDate = new Date(project.created_at * 1000).toLocaleDateString();
                
                return `
                    <div class="project-item ${isSelected ? 'selected' : ''}" 
                         data-project-name="${project.name}"
                         draggable="true"
                         ondragstart="handleDragStart(event)"
                         ondragover="handleDragOver(event)"
                         ondrop="handleDrop(event)"
                         ondragend="handleDragEnd(event)">
                        <div class="drag-handle" title="Drag to reorder">⋮⋮</div>
                        <div class="project-info">
                            <div class="project-name-container ${isSelected ? 'selected' : ''}" onclick="selectProject('${project.name}')">
                                <div class="project-name-display">${project.name}</div>
                            </div>
                            <div class="project-description-container" onclick="selectProject('${project.name}')">
                                <div class="project-description">${project.description || 'No description'}</div>
                            </div>
                            <div class="project-meta" onclick="selectProject('${project.name}')">Created: ${createdDate}</div>
                        </div>
                        <div class="project-actions-buttons">
                            <button class="btn btn-edit-project" onclick="event.stopPropagation(); editProject('${project.name}')" title="Edit Project">✏️</button>
                            <button class="btn btn-delete-project" onclick="event.stopPropagation(); deleteProject('${project.name}')" title="Delete Project">🗑️</button>
                        </div>
                        <div class="project-name-indicator">${isSelected ? '✓' : ''}</div>
                    </div>
                `;
            }).join('');
            
            projectList.innerHTML = projectItems;
        }
        
        function updateKnowledgeDatabaseInfo() {
            const projectInfoElement = document.getElementById('currentProjectInfo');
            const knowledgeContent = document.getElementById('knowledgeDatabaseContent');
            
            if (currentProject) {
                if (projectInfoElement) {
                    projectInfoElement.textContent = `Managing knowledge database for project: "${currentProject}"`;
                }
                if (knowledgeContent) {
                    knowledgeContent.classList.remove('disabled');
                }
            } else {
                if (projectInfoElement) {
                    projectInfoElement.textContent = 'Select a project to manage its knowledge database.';
                }
                if (knowledgeContent) {
                    knowledgeContent.classList.add('disabled');
                }
            }
        }
        
        async function selectProject(projectName) {
            try {
                const response = await fetch(`/api/projects/${encodeURIComponent(projectName)}/select`, {
                    method: 'POST'
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    currentProject = projectName;
                    updateCurrentProjectDisplay();
                    loadProjectList();
                    updateKnowledgeDatabaseInfo();
                    
                    // Clear localStorage categories to prevent contamination from previous project
                    localStorage.removeItem('veridoc_cache_categories');
                    localStorage.removeItem('veridoc_custom_categories');
                    
                    // Clear global category variables
                    window.cacheCategories = {};
                    window.customCategories = {};
                    
                    // Reload caches and categories for the new project
                    loadCaches();
                    loadCustomCategories();
                    
                    showNotification(`Switched to project: ${projectName}`, 'success');
                } else {
                    console.error('Error selecting project:', data.detail);
                    showNotification(`Error selecting project: ${data.detail}`, 'error');
                }
            } catch (error) {
                console.error('Error selecting project:', error);
                showNotification('Error selecting project', 'error');
            }
        }
        
        function openProjectModal(projectName = null) {
            console.log('openProjectModal called with projectName:', projectName);
            
            const modal = document.getElementById('projectModal');
            const title = document.getElementById('projectModalTitle');
            const nameInput = document.getElementById('projectName');
            const descriptionInput = document.getElementById('projectDescription');
            const saveBtn = document.getElementById('saveProjectBtn');
            
            console.log('Modal elements found:', {
                modal: !!modal,
                title: !!title,
                nameInput: !!nameInput,
                descriptionInput: !!descriptionInput,
                saveBtn: !!saveBtn
            });
            
            if (!modal) {
                console.error('Project modal not found!');
                return;
            }
            
            if (projectName) {
                // Edit mode
                editingProject = projectName;
                const project = projects[projectName];
                title.textContent = 'Edit Project';
                nameInput.value = project.name;
                descriptionInput.value = project.description || '';
                saveBtn.textContent = 'Update Project';
            } else {
                // Add mode
                editingProject = null;
                title.textContent = 'Add New Project';
                nameInput.value = '';
                descriptionInput.value = '';
                saveBtn.textContent = 'Create Project';
            }
            
            console.log('Removing hidden class from modal');
            modal.classList.remove('hidden');
            console.log('Modal classes after removal:', modal.className);
            console.log('Modal display style:', window.getComputedStyle(modal).display);
            
            // Fallback: Force display if CSS doesn't work
            if (window.getComputedStyle(modal).display === 'none') {
                console.log('CSS not working, forcing display');
                modal.style.display = 'block';
            }
            
            nameInput.focus();
        }
        
        function closeProjectModal() {
            const modal = document.getElementById('projectModal');
            modal.classList.add('hidden');
            // Reset inline style if it was set
            modal.style.display = '';
            editingProject = null;
        }
        
        function editProject(projectName) {
            openProjectModal(projectName);
        }
        
        function deleteProject(projectName) {
            console.log('deleteProject called with:', projectName);
            const modal = document.getElementById('deleteProjectModal');
            const nameSpan = document.getElementById('deleteProjectName');
            const confirmationInput = document.getElementById('deleteConfirmationInput');
            const confirmBtn = document.getElementById('confirmDeleteProjectBtn');
            
            console.log('Delete project modal found:', !!modal);
            console.log('Delete project name span found:', !!nameSpan);
            
            if (!modal) {
                console.error('Delete project modal not found!');
                return;
            }
            
            nameSpan.textContent = projectName;
            
            // Reset the confirmation input and button state
            if (confirmationInput && confirmBtn) {
                confirmationInput.value = '';
                confirmBtn.disabled = true;
                updateDeleteConfirmationStatus('');
            }
            
            console.log('Removing hidden class from delete project modal');
            modal.classList.remove('hidden');
            console.log('Modal classes after removal:', modal.className);
            console.log('Modal display style:', window.getComputedStyle(modal).display);
            
            // Focus on the confirmation input
            if (confirmationInput) {
                setTimeout(() => {
                    confirmationInput.focus();
                }, 100);
            }
            
            // Fallback: Force display if CSS doesn't work
            if (window.getComputedStyle(modal).display === 'none') {
                console.log('CSS not working, forcing display for delete project modal');
                modal.style.display = 'block';
            }
        }
        
        function checkDeleteConfirmation() {
            const input = document.getElementById('deleteConfirmationInput');
            const confirmBtn = document.getElementById('confirmDeleteProjectBtn');
            const statusDiv = document.getElementById('deleteConfirmationStatus');
            
            if (!input || !confirmBtn || !statusDiv) return;
            
            const inputValue = input.value.trim().toLowerCase();
            const isCorrect = inputValue === 'delete';
            
            confirmBtn.disabled = !isCorrect;
            
            if (inputValue === '') {
                updateDeleteConfirmationStatus('');
            } else if (isCorrect) {
                updateDeleteConfirmationStatus('✅ Correct! You can now delete the project.', 'success');
            } else {
                updateDeleteConfirmationStatus('❌ Incorrect. Please type "delete" exactly.', 'error');
            }
        }
        
        function updateDeleteConfirmationStatus(message, type = '') {
            const statusDiv = document.getElementById('deleteConfirmationStatus');
            if (!statusDiv) return;
            
            statusDiv.textContent = message;
            statusDiv.className = '';
            
            if (type === 'success') {
                statusDiv.style.color = '#059669';
                statusDiv.style.fontWeight = '600';
            } else if (type === 'error') {
                statusDiv.style.color = '#dc2626';
                statusDiv.style.fontWeight = '600';
            } else {
                statusDiv.style.color = '#6b7280';
                statusDiv.style.fontWeight = 'normal';
            }
        }
        
        function closeDeleteProjectModal() {
            const modal = document.getElementById('deleteProjectModal');
            modal.classList.add('hidden');
            // Reset inline style if it was set
            modal.style.display = '';
        }
        
        async function saveProject() {
            const nameInput = document.getElementById('projectName');
            const descriptionInput = document.getElementById('projectDescription');
            
            const projectName = nameInput.value.trim();
            const description = descriptionInput.value.trim();
            
            if (!projectName) {
                showNotification('Project name is required', 'error');
                return;
            }
            
            try {
                let response;
                if (editingProject) {
                    // Update existing project
                    response = await fetch(`/api/projects/${encodeURIComponent(editingProject)}`, {
                        method: 'PUT',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            name: projectName,
                            description: description
                        })
                    });
                } else {
                    // Create new project
                    response = await fetch('/api/projects', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({
                            name: projectName,
                            description: description
                        })
                    });
                }
                
                const data = await response.json();
                
                if (response.ok) {
                    projects = data.projects;
                    currentProject = data.current_project;
                    
                    updateCurrentProjectDisplay();
                    loadProjectList();
                    updateKnowledgeDatabaseInfo();
                    
                    closeProjectModal();
                    
                    const action = editingProject ? 'updated' : 'created';
                    showNotification(`Project "${projectName}" ${action} successfully`, 'success');
                } else {
                    console.error('Error saving project:', data.detail);
                    showNotification(`Error saving project: ${data.detail}`, 'error');
                }
            } catch (error) {
                console.error('Error saving project:', error);
                showNotification('Error saving project', 'error');
            }
        }
        
        async function confirmDeleteProject() {
            const nameSpan = document.getElementById('deleteProjectName');
            const projectName = nameSpan.textContent;
            
            try {
                const response = await fetch(`/api/projects/${encodeURIComponent(projectName)}`, {
                    method: 'DELETE'
                });
                
                const data = await response.json();
                
                if (response.ok) {
                    projects = data.projects;
                    currentProject = data.current_project;
                    
                    updateCurrentProjectDisplay();
                    loadProjectList();
                    updateKnowledgeDatabaseInfo();
                    
                    closeDeleteProjectModal();
                    
                    showNotification(`Project "${projectName}" deleted successfully`, 'success');
                } else {
                    console.error('Error deleting project:', data.detail);
                    showNotification(`Error deleting project: ${data.detail}`, 'error');
                }
            } catch (error) {
                console.error('Error deleting project:', error);
                showNotification('Error deleting project', 'error');
            }
        }
        
        function toggleProjectReorderMode() {
            const projectList = document.getElementById('projectList');
            const reorderBtn = document.getElementById('reorderProjectsBtn');
            
            if (projectList.classList.contains('reordering')) {
                // Exit reorder mode
                projectList.classList.remove('reordering');
                reorderBtn.textContent = '🔄 Reorder';
                reorderBtn.classList.remove('active');
                
                // Save the new order if it changed
                saveProjectOrder();
            } else {
                // Enter reorder mode
                projectList.classList.add('reordering');
                reorderBtn.textContent = '✅ Done';
                reorderBtn.classList.add('active');
            }
        }
        
        // Drag and Drop Functions
        let draggedElement = null;
        let draggedIndex = -1;
        
        function handleDragStart(event) {
            const projectList = document.getElementById('projectList');
            if (!projectList.classList.contains('reordering')) {
                event.preventDefault();
                return;
            }
            
            draggedElement = event.target;
            draggedElement.classList.add('dragging');
            
            // Find the index of the dragged element
            const projectItems = Array.from(document.querySelectorAll('.project-item'));
            draggedIndex = projectItems.indexOf(draggedElement);
            
            event.dataTransfer.effectAllowed = 'move';
            event.dataTransfer.setData('text/html', draggedElement.outerHTML);
        }
        
        function handleDragOver(event) {
            const projectList = document.getElementById('projectList');
            if (!projectList.classList.contains('reordering')) {
                return;
            }
            
            event.preventDefault();
            event.dataTransfer.dropEffect = 'move';
            
            const afterElement = getDragAfterElement(projectList, event.clientY);
            const dragging = document.querySelector('.dragging');
            
            if (afterElement == null) {
                projectList.appendChild(dragging);
            } else {
                projectList.insertBefore(dragging, afterElement);
            }
        }
        
        function handleDrop(event) {
            event.preventDefault();
            return false;
        }
        
        function handleDragEnd(event) {
            const projectList = document.getElementById('projectList');
            if (!projectList.classList.contains('reordering')) {
                return;
            }
            
            if (draggedElement) {
                draggedElement.classList.remove('dragging');
                draggedElement = null;
                draggedIndex = -1;
            }
        }
        
        function getDragAfterElement(container, y) {
            const draggableElements = [...container.querySelectorAll('.project-item:not(.dragging)')];
            
            return draggableElements.reduce((closest, child) => {
                const box = child.getBoundingClientRect();
                const offset = y - box.top - box.height / 2;
                
                if (offset < 0 && offset > closest.offset) {
                    return { offset: offset, element: child };
                } else {
                    return closest;
                }
            }, { offset: Number.NEGATIVE_INFINITY }).element;
        }
        
        async function saveProjectOrder() {
            try {
                const projectItems = Array.from(document.querySelectorAll('.project-item'));
                const projectOrder = projectItems.map(item => item.getAttribute('data-project-name'));
                
                const response = await fetch('/api/projects/reorder', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json; charset=utf-8'
                    },
                    body: JSON.stringify({
                        order: projectOrder
                    })
                });
                
                if (response.ok) {
                    const data = await response.json();
                    projects = data.projects;
                    showNotification('Project order updated successfully', 'success');
                } else {
                    throw new Error('Failed to save project order');
                }
            } catch (error) {
                console.error('Error saving project order:', error);
                showNotification('Failed to save project order', 'error');
            }
        }
        
        function ensureAllButtonsWorking() {
            console.log('Ensuring all buttons are working...');
            
            // Check project management buttons
            const projectButtons = ['addProjectBtn', 'reorderProjectsBtn', 'saveProjectBtn', 'confirmDeleteProjectBtn'];
            projectButtons.forEach(buttonId => {
                const button = document.getElementById(buttonId);
                if (button) {
                    console.log(`✓ ${buttonId} found`);
                } else {
                    console.log(`✗ ${buttonId} not found`);
                }
            });
            
            // Check knowledge database buttons
            const knowledgeButtons = ['manageCategoriesBtn', 'bulkAssignBtn', 'loadCachesBtn', 'syncCategoriesBtn', 'deleteCachesBtn'];
            knowledgeButtons.forEach(buttonId => {
                const button = document.getElementById(buttonId);
                if (button) {
                    console.log(`✓ ${buttonId} found`);
                } else {
                    console.log(`✗ ${buttonId} not found`);
                }
            });
        }
        
        function retryButtonInitialization() {
            console.log('Retrying button initialization...');
            
            // Try to find and initialize buttons that might not have been found initially
            const addProjectBtn = document.getElementById('addProjectBtn');
            if (addProjectBtn && !addProjectBtn.hasAttribute('data-listener-added')) {
                console.log('Adding event listener to Add Project button');
                addProjectBtn.addEventListener('click', () => {
                    console.log('Add Project button clicked (retry)');
                    openProjectModal();
                });
                addProjectBtn.setAttribute('data-listener-added', 'true');
            }
            
            const manageCategoriesBtn = document.getElementById('manageCategoriesBtn');
            if (manageCategoriesBtn && !manageCategoriesBtn.hasAttribute('data-listener-added')) {
                console.log('Adding event listener to Manage Categories button');
                manageCategoriesBtn.addEventListener('click', () => {
                    console.log('Manage Categories button clicked (retry)');
                    openCategoryModal();
                });
                manageCategoriesBtn.setAttribute('data-listener-added', 'true');
            }
            
            const bulkAssignBtn = document.getElementById('bulkAssignBtn');
            if (bulkAssignBtn && !bulkAssignBtn.hasAttribute('data-listener-added')) {
                console.log('Adding event listener to Bulk Assign button');
                bulkAssignBtn.addEventListener('click', () => {
                    console.log('Bulk Assign button clicked (retry)');
                    openBulkModal();
                });
                bulkAssignBtn.setAttribute('data-listener-added', 'true');
            }
        }
        
        function showNotification(message, type = 'info') {
            // Create notification element
            const notification = document.createElement('div');
            notification.className = `notification notification-${type}`;
            notification.textContent = message;
            
            // Style the notification
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                padding: 12px 20px;
                border-radius: 6px;
                color: white;
                font-weight: 500;
                z-index: 10000;
                animation: slideIn 0.3s ease-out;
                max-width: 400px;
                word-wrap: break-word;
            `;
            
            // Set background color based on type
            switch (type) {
                case 'success':
                    notification.style.backgroundColor = '#10b981';
                    break;
                case 'error':
                    notification.style.backgroundColor = '#ef4444';
                    break;
                case 'warning':
                    notification.style.backgroundColor = '#f59e0b';
                    break;
                default:
                    notification.style.backgroundColor = '#3b82f6';
            }
            
            // Add to page
            document.body.appendChild(notification);
            
            // Remove after 5 seconds
            setTimeout(() => {
                notification.style.animation = 'slideOut 0.3s ease-in';
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 300);
            }, 5000);
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

@app.get("/mobile", response_class=HTMLResponse)
async def get_mobile_page():
    """Serve the mobile HTML page - automatically accessed via server-side detection"""
    html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VeriDoc AI - Mobile</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="mobile-container">
        <div class="mobile-header">
            <h1>VeriDoc AI</h1>
            <p>Mobile Interface</p>
            <div id="mobileCurrentProjectDisplay" class="mobile-current-project-display">
                <span class="mobile-project-label">Project:</span>
                <span id="mobileCurrentProjectName" class="mobile-project-name">Loading...</span>
            </div>
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
        let mobileCurrentProject = null;
        
        // Initialize mobile interface
        document.addEventListener('DOMContentLoaded', function() {
            loadMobileProjects();
            loadMobileKnowledgeDatabase();
            setupMobileEventListeners();
        });
        
        async function loadMobileProjects() {
            try {
                const response = await fetch('/api/projects');
                const data = await response.json();
                
                if (response.ok) {
                    mobileCurrentProject = data.current_project;
                    
                    // Update mobile header display
                    const projectNameElement = document.getElementById('mobileCurrentProjectName');
                    if (projectNameElement) {
                        projectNameElement.textContent = mobileCurrentProject || 'No Project Selected';
                    }
                } else {
                    console.error('Error loading projects:', data.detail);
                }
            } catch (error) {
                console.error('Error loading projects:', error);
            }
        }
        
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
            const messagesContainer = document.getElementById('mobileConversationMessages');
            
            if (!messagesContainer) {
                console.error('mobileConversationMessages container not found');
                alert('Error: Conversation container not found');
                return;
            }
            
            const messages = messagesContainer.querySelectorAll('.mobile-message');
            
            if (messages.length === 0) {
                alert('No conversation to copy');
                return;
            }
            
            let conversationText = '';
            
            messages.forEach((message, index) => {
                const role = message.classList.contains('user') ? 'User' : 'Assistant';
                const time = message.querySelector('.timestamp')?.textContent || '';
                const content = message.querySelector('.content')?.textContent || '';
                
                conversationText += `[${time}] ${role}:\n${content}\n\n`;
            });
            
            // Remove trailing newlines
            conversationText = conversationText.trim();
            
            const copyBtn = document.querySelector('.mobile-conversation-copy-btn');
            
            if (!copyBtn) {
                console.error('Copy button not found');
                alert('Error: Copy button not found');
                return;
            }
            
            // Try modern clipboard API first
            if (navigator.clipboard && navigator.clipboard.writeText) {
                navigator.clipboard.writeText(conversationText).then(() => {
                                            // Visual feedback
                        copyBtn.classList.add('copied');
                        copyBtn.textContent = '✓';
                        
                        setTimeout(() => {
                            copyBtn.classList.remove('copied');
                            copyBtn.textContent = '📋';
                        }, 2000);
                    }).catch(err => {
                        copyWithFallback(conversationText, copyBtn);
                    });
            } else {
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
    """Start the quick analysis process with proper validation"""
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
        verification_params = data.get("verification_params", {})
        output_fields = data.get("output_fields", {})
        output_field_descriptions = data.get("output_field_descriptions", {})
        
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
        
        logger.info(f"Starting quick analysis process with {len(selected_caches)} caches and {len(knowledge_files)} files")
        
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
            selected_statements,
            verification_params,
            output_fields,
            output_field_descriptions
        )
        
        return {"message": "Processing started successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting processing: {e}")
        raise HTTPException(status_code=500, detail="Failed to start processing")

@app.post("/api/validate-excel")
async def validate_excel_file(request: Request):
    """Validate Excel file structure before processing"""
    try:
        data = await request.json()
        statements_file = data.get("statements_file", "")
        
        if not statements_file:
            raise HTTPException(status_code=400, detail="No statements file specified")
        
        statements_path = UPLOAD_DIR / statements_file
        if not statements_path.exists():
            raise HTTPException(status_code=404, detail=f"Statements file '{statements_file}' not found")
        
        try:
            # Try to read the statements to validate structure
            all_statements = read_statements(str(statements_path))
            
            # If we get here, the file is valid
            return {
                "valid": True,
                "message": f"Excel file is valid. Found {len(all_statements)} statements.",
                "statement_count": len(all_statements),
                "file_info": {
                    "filename": statements_file,
                    "columns_used": "First 2 columns (position-based)",
                    "additional_columns_ignored": True
                }
            }
            
        except Exception as e:
            # File structure is invalid
            return {
                "valid": False,
                "message": f"Excel file validation failed: {str(e)}",
                "error_details": str(e),
                "requirements": {
                    "min_columns": 2,
                    "column_1_purpose": "Paragraph number (required, cannot be empty)",
                    "column_2_purpose": "Statement content (required, cannot be empty)",
                    "additional_columns": "Will be ignored"
                }
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error validating Excel file: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate Excel file")

@app.get("/api/caches")
async def get_caches():
    """Get list of available caches for the current project"""
    try:
        # Use project-specific cache directory
        project_cache_dir = get_project_cache_dir(current_project) if current_project else ".cache"
        
        # Check if project-specific cache directory exists and has files
        if current_project and not project_cache_dir.exists():
            # Try to migrate existing caches from .index_cache
            migrate_existing_caches_to_project(current_project)
        
        caches = list_available_caches(str(project_cache_dir))
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

# Project Management Functions
def load_projects():
    """Load projects from file"""
    global projects, current_project
    try:
        if PROJECTS_FILE.exists():
            with open(PROJECTS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                projects = data.get('projects', {})
                current_project = data.get('current_project', None)
                
                # If no current project is set, set the first one as current
                if not current_project and projects:
                    current_project = list(projects.keys())[0]
                    save_projects()
        else:
            # Create default project
            default_project_name = "Default Project"
            projects = {
                default_project_name: {
                    "name": default_project_name,
                    "created_at": time.time(),
                    "description": "Default project for VeriDoc AI"
                }
            }
            current_project = default_project_name
            save_projects()
            
        logger.info(f"Loaded {len(projects)} projects, current: {current_project}")
    except Exception as e:
        logger.error(f"Error loading projects: {e}")
        # Create default project on error
        default_project_name = "Default Project"
        projects = {
            default_project_name: {
                "name": default_project_name,
                "created_at": time.time(),
                "description": "Default project for VeriDoc AI"
            }
        }
        current_project = default_project_name
        save_projects()

def save_projects():
    """Save projects to file"""
    try:
        data = {
            "projects": projects,
            "current_project": current_project
        }
        with open(PROJECTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(projects)} projects")
    except Exception as e:
        logger.error(f"Error saving projects: {e}")

def get_project_categories_file(project_name: str):
    """Get the categories file path for a specific project"""
    return Path(f"config/project_categories_{project_name.replace(' ', '_').lower()}.json")

def get_project_cache_dir(project_name: str):
    """Get the cache directory path for a specific project"""
    safe_name = project_name.replace(' ', '_').lower()
    return Path("caches") / f"cache_{safe_name}"

def get_project_ask_me_cache_dir(project_name: str):
    """Get the Ask Me cache directory path for a specific project"""
    safe_name = project_name.replace(' ', '_').lower()
    return Path("caches") / f"ask_me_cache_{safe_name}"

def migrate_existing_caches_to_project(project_name: str):
    """Migrate existing caches from old cache directories to new caches folder structure"""
    try:
        # Ensure the main caches directory exists
        main_caches_dir = Path("caches")
        main_caches_dir.mkdir(exist_ok=True)
        
        new_cache_dir = get_project_cache_dir(project_name)
        new_ask_me_cache_dir = get_project_ask_me_cache_dir(project_name)
        
        if new_cache_dir.exists() and new_ask_me_cache_dir.exists():
            logger.info(f"Project cache directories already exist for {project_name}")
            return
        
        # Check if this is an existing project that should have caches migrated
        categories_file = get_project_categories_file(project_name)
        if categories_file.exists():
            try:
                with open(categories_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    existing_categories = data.get('categories', {})
                    # If there are existing categories, this is likely an existing project that should be migrated
                    if existing_categories:
                        logger.info(f"Migrating existing caches for project {project_name} (has existing categories)")
                    else:
                        logger.info(f"Project {project_name} has no existing categories, creating empty cache directories")
                        new_cache_dir.mkdir(parents=True, exist_ok=True)
                        new_ask_me_cache_dir.mkdir(parents=True, exist_ok=True)
                        return
            except Exception as e:
                logger.warning(f"Could not read categories file for {project_name}: {e}")
                # If we can't read the categories file, assume it's a new project
                logger.info(f"Creating empty cache directories for project {project_name}")
                new_cache_dir.mkdir(parents=True, exist_ok=True)
                new_ask_me_cache_dir.mkdir(parents=True, exist_ok=True)
                return
        else:
            # No categories file exists, this is definitely a new project
            logger.info(f"Creating empty cache directories for new project {project_name}")
            new_cache_dir.mkdir(parents=True, exist_ok=True)
            new_ask_me_cache_dir.mkdir(parents=True, exist_ok=True)
            return
        
        # Create the new project-specific cache directories
        new_cache_dir.mkdir(parents=True, exist_ok=True)
        new_ask_me_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Migrate from old individual cache directories
        old_cache_dir = Path(f".cache_{project_name.replace(' ', '_').lower()}")
        old_ask_me_cache_dir = Path(f".ask_me_cache_{project_name.replace(' ', '_').lower()}")
        
        import shutil
        
        # Migrate main cache directory
        if old_cache_dir.exists():
            cache_files = [f for f in old_cache_dir.iterdir() if f.is_file()]
            for file_path in cache_files:
                shutil.copy2(file_path, new_cache_dir / file_path.name)
            logger.info(f"Migrated {len(cache_files)} cache files from {old_cache_dir} to {new_cache_dir}")
        
        # Migrate Ask Me cache directory
        if old_ask_me_cache_dir.exists():
            ask_me_files = [f for f in old_ask_me_cache_dir.iterdir() if f.is_file()]
            for file_path in ask_me_files:
                shutil.copy2(file_path, new_ask_me_cache_dir / file_path.name)
            logger.info(f"Migrated {len(ask_me_files)} Ask Me cache files from {old_ask_me_cache_dir} to {new_ask_me_cache_dir}")
        
        # Also check for old .index_cache directory (legacy)
        old_index_cache_dir = Path(".index_cache")
        if old_index_cache_dir.exists() and not any(new_cache_dir.iterdir()):
            cache_files = [f for f in old_index_cache_dir.iterdir() if f.is_file()]
            for file_path in cache_files:
                shutil.copy2(file_path, new_cache_dir / file_path.name)
            logger.info(f"Migrated {len(cache_files)} cache files from legacy {old_index_cache_dir} to {new_cache_dir}")
        
    except Exception as e:
        logger.error(f"Error migrating caches for project {project_name}: {e}")

def load_project_categories(project_name: str):
    """Load categories for a specific project"""
    try:
        categories_file = get_project_categories_file(project_name)
        if categories_file.exists():
            with open(categories_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('categories', {}), data.get('all_categories', DEFAULT_CATEGORIES.copy())
        else:
            # Return default categories for new projects
            return {}, DEFAULT_CATEGORIES.copy()
    except Exception as e:
        logger.error(f"Error loading project categories for {project_name}: {e}")
        return {}, DEFAULT_CATEGORIES.copy()

def save_project_categories(project_name: str, categories: dict, all_categories_list: list):
    """Save categories for a specific project"""
    try:
        categories_file = get_project_categories_file(project_name)
        data = {
            "categories": categories,
            "all_categories": all_categories_list
        }
        with open(categories_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved categories for project: {project_name}")
    except Exception as e:
        logger.error(f"Error saving project categories for {project_name}: {e}")

# Initialize projects and categories after function definitions
load_projects()
load_cache_categories()

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

def find_latest_results_file():
    """Find the latest verification results file (standard or timestamped)"""
    import glob
    import os
    
    # Collect all possible files with their modification times
    possible_files = []
    
    # Check for standard file
    if os.path.exists("verification_results.xlsx"):
        mtime = os.path.getmtime("verification_results.xlsx")
        possible_files.append(("verification_results.xlsx", mtime, "standard"))
    
    # Check for timestamped files
    timestamped_files = glob.glob("verification_results_*.xlsx")
    for file in timestamped_files:
        try:
            mtime = os.path.getmtime(file)
            possible_files.append((file, mtime, "timestamped"))
        except Exception as e:
            logger.warning(f"Could not get modification time for {file}: {e}")
            continue
    
    # Check for temp files
    temp_files = glob.glob("temp_verification_results_*.xlsx")
    for file in temp_files:
        try:
            mtime = os.path.getmtime(file)
            possible_files.append((file, mtime, "temporary"))
        except Exception as e:
            logger.warning(f"Could not get modification time for {file}: {e}")
            continue
    
    if not possible_files:
        return None
    
    # Sort by modification time (newest first) - this ensures we get the most recent file
    possible_files.sort(key=lambda x: x[1], reverse=True)
    
    latest_file = possible_files[0][0]
    latest_mtime = possible_files[0][1]
    latest_type = possible_files[0][2]
    
    logger.info(f"Found latest results file: {latest_file}")
    

    
    return latest_file

@app.get("/api/results")
async def get_results(filename: str = None):
    """Get verification results from a specific file or the latest available file"""
    try:
        # If filename is specified, use it; otherwise find the latest
        if filename:
            if not os.path.exists(filename):
                raise HTTPException(status_code=404, detail=f"File {filename} not found")
            results_file = filename
            logger.info(f"Loading results from specified file: {results_file}")
        else:
            # Find the latest results file
            results_file = find_latest_results_file()
            if not results_file:
                logger.info("No verification results files found")
                return []
            logger.info(f"Loading results from latest file: {results_file}")
        
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
        
        # Sort results by paragraph number
        def sort_by_paragraph_number(result):
            # Try to get paragraph number from various possible column names
            par_number = result.get('Paragraph Number') or result.get('Par Number') or result.get('Paragraph') or ''
            
            # Handle different paragraph number formats
            if isinstance(par_number, str):
                # Remove any non-alphanumeric characters and try to extract numbers
                import re
                numbers = re.findall(r'\d+', par_number)
                if numbers:
                    # Use the first number found for sorting
                    return (0, int(numbers[0]))  # Tuple: (0 = numeric, value)
                # If no numbers found, try to sort alphabetically
                return (1, par_number.lower())  # Tuple: (1 = alphabetical, value)
            elif isinstance(par_number, (int, float)):
                return (0, float(par_number))  # Tuple: (0 = numeric, value)
            elif par_number is None or par_number == '':
                # If paragraph number is None or empty, put at the end
                return (2, float('inf'))  # Tuple: (2 = invalid, infinity)
            else:
                # If paragraph number is invalid, put at the end
                return (2, float('inf'))  # Tuple: (2 = invalid, infinity)
        
        # Sort the cleaned data by paragraph number
        cleaned_data.sort(key=sort_by_paragraph_number)
        
        logger.info(f"Successfully loaded and sorted {len(cleaned_data)} results from {results_file}")
        return cleaned_data
    except Exception as e:
        logger.error(f"Failed to fetch results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch results: {str(e)}")

@app.get("/api/results/download")
async def download_results():
    """Download verification results from the latest available file"""
    try:
        # Find the latest results file
        results_file = find_latest_results_file()
        if not results_file:
            raise HTTPException(status_code=404, detail="No verification results files found")
        
        # Extract filename for download
        download_filename = os.path.basename(results_file)
        
        logger.info(f"Downloading results from: {results_file}")
        
        return FileResponse(
            results_file,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=download_filename
        )
    except Exception as e:
        logger.error(f"Failed to download results: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to download results: {str(e)}")

@app.get("/api/results/files")
async def get_results_files():
    """Get information about available verification results files"""
    try:
        import glob
        import os
        
        files_info = []
        
        # Check for standard file
        if os.path.exists("verification_results.xlsx"):
            mtime = os.path.getmtime("verification_results.xlsx")
            files_info.append({
                "filename": "verification_results.xlsx",
                "type": "standard",
                "modified": mtime,
                "size": os.path.getsize("verification_results.xlsx")
            })
        
        # Check for timestamped files
        timestamped_files = glob.glob("verification_results_*.xlsx")
        for file in timestamped_files:
            try:
                mtime = os.path.getmtime(file)
                size = os.path.getsize(file)
                files_info.append({
                    "filename": file,
                    "type": "timestamped",
                    "modified": mtime,
                    "size": size
                })
            except Exception as e:
                logger.warning(f"Could not get info for file {file}: {e}")
        
        # Check for temp files
        temp_files = glob.glob("temp_verification_results_*.xlsx")
        for file in temp_files:
            try:
                mtime = os.path.getmtime(file)
                size = os.path.getsize(file)
                files_info.append({
                    "filename": file,
                    "type": "temporary",
                    "modified": mtime,
                    "size": size
                })
            except Exception as e:
                logger.warning(f"Could not get info for file {file}: {e}")
        
        # Sort by modification time (newest first)
        files_info.sort(key=lambda x: x["modified"], reverse=True)
        
        # Add human-readable information
        for file_info in files_info:
            file_info["modified_readable"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(file_info["modified"]))
            file_info["size_mb"] = round(file_info["size"] / (1024 * 1024), 2)
        
        return {
            "files": files_info,
            "total_files": len(files_info),
            "latest_file": files_info[0]["filename"] if files_info else None
        }
        
    except Exception as e:
        logger.error(f"Failed to get results files info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get results files info: {str(e)}")



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
         
         # Debug logging
         logger.info(f"Loaded {len(statements)} statements from {filename}")
         if statements:
             logger.info(f"First statement type: {type(statements[0])}")
             logger.info(f"First statement keys: {statements[0].keys() if isinstance(statements[0], dict) else 'Not a dict'}")
         
         return {"statements": statements}
     except Exception as e:
         logger.error(f"Error in get_statements_data: {str(e)}")
         import traceback
         logger.error(f"Traceback: {traceback.format_exc()}")
         raise HTTPException(status_code=500, detail=f"Failed to load statements data: {str(e)}")

@app.post("/api/save-output-fields")
async def save_output_fields(request: Request):
    """Save output field configurations to server with validation"""
    try:
        data = await request.json()
        fields = data.get("fields", [])
        
        # Validation: Check for fixed columns that cannot be deleted or renamed
        fixed_column_ids = ["par_number", "par_context"]
        fixed_column_names = ["Paragraph Number", "Statement"]
        
        # Ensure fixed columns exist and have correct names
        for i, fixed_id in enumerate(fixed_column_ids):
            fixed_field = next((f for f in fields if f.get("id") == fixed_id), None)
            if not fixed_field:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Fixed column '{fixed_id}' cannot be deleted. It must always exist."
                )
            if fixed_field.get("name") != fixed_column_names[i]:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Fixed column '{fixed_id}' cannot be renamed. It must remain as '{fixed_column_names[i]}'."
                )
            if not fixed_field.get("enabled", True):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Fixed column '{fixed_id}' cannot be disabled. It must always be enabled."
                )
        
        # Validation: Check for duplicate field names
        field_names = [f.get("name", "") for f in fields if f.get("enabled", True)]
        duplicate_names = [name for name in field_names if field_names.count(name) > 1]
        if duplicate_names:
            raise HTTPException(
                status_code=400, 
                detail=f"Duplicate field names found: {', '.join(set(duplicate_names))}. Each field must have a unique name."
            )
        
        # Save to a JSON file
        output_fields_file = "output_fields_config.json"
        with open(output_fields_file, 'w', encoding='utf-8') as f:
            json.dump(fields, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(fields)} output fields to server")
        return {"message": "Output fields saved successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error saving output fields: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save output fields: {str(e)}")

@app.get("/api/load-output-fields")
async def load_output_fields():
    """Load output field configurations from server"""
    try:
        output_fields_file = "output_fields_config.json"
        if os.path.exists(output_fields_file):
            with open(output_fields_file, 'r', encoding='utf-8') as f:
                fields = json.load(f)
            return {"fields": fields}
        else:
            # Return default fields if no saved configuration exists
            default_fields = [
                {"id": "par_number", "name": "Paragraph Number from the input file", "description": "", "enabled": True, "fixed": True},
                {"id": "par_context", "name": "Statement content from the input file", "description": "", "enabled": True, "fixed": True},
                {"id": "is_accurate", "name": "Document Reference", "description": "List documents which are mentioned in the statement", "enabled": True, "fixed": False},
                {"id": "field_1756803554789", "name": "Fact/Finding", "description": "Check the statement below if there are Facts or Findings in respect of Primecap only in this particular statement? (Answer short Fact/Finding))", "enabled": True, "fixed": False},
                {"id": "field_1756803586927", "name": "Findings", "description": "List the findings in respect of Primecap in (a),(b),(c) list format", "enabled": True, "fixed": False}
            ]
            return {"fields": default_fields}
    except Exception as e:
        logger.error(f"Error loading output fields: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load output fields: {str(e)}")

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
                
                # Build or load index for this file using project-specific cache directory
                project_cache_dir = get_project_cache_dir(current_project) if current_project else ".cache"
                os.makedirs(project_cache_dir, exist_ok=True)
                
                embeddings, chunks, index_id = build_or_load_index(
                    corpus_path=str(file_path),
                    embedding_model="text-embedding-3-large",
                    client=client,
                    chunk_size_chars=4000,
                    overlap_chars=500,
                    cache_dir=str(project_cache_dir)
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
        
        # Load knowledge database caches with project-specific cache directory
        if selected_caches:
            project_cache_dir = get_project_cache_dir(current_project) if current_project else ".cache"
            
            # Check if project-specific cache directory exists and migrate if needed
            if current_project and not project_cache_dir.exists():
                migrate_existing_caches_to_project(current_project)
            
            embeddings, chunks, cache_names = load_multiple_caches(selected_caches, str(project_cache_dir))
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



@app.get("/api/status")
async def get_processing_status():
    """Get current processing status"""
    from utils.status import get_processing_status
    return get_processing_status()

# Project Management API Endpoints
@app.get("/api/projects")
async def get_projects():
    """Get all projects and current project"""
    return {
        "projects": projects,
        "current_project": current_project
    }

@app.post("/api/projects")
async def create_project(request: Request):
    """Create a new project"""
    try:
        global projects, current_project
        
        data = await request.json()
        project_name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        
        if not project_name:
            raise HTTPException(status_code=400, detail="Project name is required")
        
        if project_name in projects:
            raise HTTPException(status_code=400, detail="Project name already exists")
        
        # Create new project
        projects[project_name] = {
            "name": project_name,
            "description": description,
            "created_at": time.time()
        }
        
        # Set as current project if it's the first one
        if not current_project:
            current_project = project_name
        
        save_projects()
        
        # Initialize empty categories for the new project
        save_project_categories(project_name, {}, DEFAULT_CATEGORIES.copy())
        
        # Create empty cache directories for the new project in caches folder
        project_cache_dir = get_project_cache_dir(project_name)
        project_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created empty cache directory for new project: {project_cache_dir}")
        
        # Create empty Ask Me cache directory for the new project in caches folder
        project_ask_me_cache_dir = get_project_ask_me_cache_dir(project_name)
        project_ask_me_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created empty Ask Me cache directory for new project: {project_ask_me_cache_dir}")
        
        logger.info(f"Created new project: {project_name}")
        return {
            "message": f"Project '{project_name}' created successfully",
            "projects": projects,
            "current_project": current_project
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create project: {str(e)}")

@app.put("/api/projects/{project_name}")
async def update_project(project_name: str, request: Request):
    """Update project name or description"""
    try:
        global projects, current_project
        
        if project_name not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        data = await request.json()
        new_name = data.get("name", "").strip()
        description = data.get("description", "").strip()
        
        if not new_name:
            raise HTTPException(status_code=400, detail="Project name is required")
        
        if new_name != project_name and new_name in projects:
            raise HTTPException(status_code=400, detail="Project name already exists")
        
        # Update project
        old_project_data = projects[project_name]
        old_project_data["name"] = new_name
        old_project_data["description"] = description
        
        # If name changed, update the key
        if new_name != project_name:
            projects[new_name] = old_project_data
            del projects[project_name]
            
            # Update current project if it was the renamed one
            if current_project == project_name:
                current_project = new_name
            
            # Rename project categories file
            old_categories_file = get_project_categories_file(project_name)
            new_categories_file = get_project_categories_file(new_name)
            if old_categories_file.exists():
                old_categories_file.rename(new_categories_file)
        
        save_projects()
        
        logger.info(f"Updated project: {project_name} -> {new_name}")
        return {
            "message": f"Project updated successfully",
            "projects": projects,
            "current_project": current_project
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update project: {str(e)}")

@app.delete("/api/projects/{project_name}")
async def delete_project(project_name: str):
    """Delete a project and its associated data"""
    try:
        global projects, current_project
        
        if project_name not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        if len(projects) <= 1:
            raise HTTPException(status_code=400, detail="Cannot delete the last project")
        
        # Delete project categories file
        categories_file = get_project_categories_file(project_name)
        if categories_file.exists():
            categories_file.unlink()
        
        # Delete project-specific cache directories from caches folder
        cache_dir = get_project_cache_dir(project_name)
        ask_me_cache_dir = get_project_ask_me_cache_dir(project_name)
        
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            logger.info(f"Deleted project cache directory: {cache_dir}")
        
        if ask_me_cache_dir.exists():
            import shutil
            shutil.rmtree(ask_me_cache_dir)
            logger.info(f"Deleted project Ask Me cache directory: {ask_me_cache_dir}")
        
        # Also clean up old individual cache directories if they exist
        old_cache_dir = Path(f".cache_{project_name.replace(' ', '_').lower()}")
        old_ask_me_cache_dir = Path(f".ask_me_cache_{project_name.replace(' ', '_').lower()}")
        
        if old_cache_dir.exists():
            import shutil
            shutil.rmtree(old_cache_dir)
            logger.info(f"Deleted old project cache directory: {old_cache_dir}")
        
        if old_ask_me_cache_dir.exists():
            import shutil
            shutil.rmtree(old_ask_me_cache_dir)
            logger.info(f"Deleted old project Ask Me cache directory: {old_ask_me_cache_dir}")
        
        # Remove project
        del projects[project_name]
        
        # Update current project if it was the deleted one
        if current_project == project_name:
            current_project = list(projects.keys())[0]
        
        save_projects()
        
        logger.info(f"Deleted project: {project_name}")
        return {
            "message": f"Project '{project_name}' deleted successfully",
            "projects": projects,
            "current_project": current_project
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete project: {str(e)}")

@app.post("/api/projects/{project_name}/select")
async def select_project(project_name: str):
    """Select a project as the current active project"""
    try:
        if project_name not in projects:
            raise HTTPException(status_code=404, detail="Project not found")
        
        global current_project
        current_project = project_name
        save_projects()
        
        # Switch to project-specific categories
        switch_to_project_categories(project_name)
        
        logger.info(f"Selected project: {project_name}")
        return {
            "message": f"Project '{project_name}' selected successfully",
            "current_project": current_project,
            "categories": all_categories
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error selecting project: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to select project: {str(e)}")

@app.post("/api/projects/reorder")
async def reorder_projects(request: Request):
    """Reorder projects"""
    try:
        global projects
        
        data = await request.json()
        project_order = data.get("order", [])
        
        if not project_order:
            raise HTTPException(status_code=400, detail="Project order is required")
        
        # Validate that all projects are included
        if set(project_order) != set(projects.keys()):
            raise HTTPException(status_code=400, detail="Invalid project order")
        
        # Reorder projects
        new_projects = {}
        for project_name in project_order:
            new_projects[project_name] = projects[project_name]
        
        projects = new_projects
        save_projects()
        
        logger.info("Projects reordered successfully")
        return {
            "message": "Projects reordered successfully",
            "projects": projects
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reordering projects: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reorder projects: {str(e)}")

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
        cache_dir = get_project_ask_me_cache_dir(current_project) if current_project else ".ask_me_cache"
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
            cache_dir = get_project_ask_me_cache_dir(current_project) if current_project else ".ask_me_cache"
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
    
    # Set up asyncio event loop policy to handle connection errors
    if sys.platform == "win32":
        # Windows-specific asyncio configuration
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Suppress asyncio connection reset warnings
    import logging
    asyncio_logger = logging.getLogger('asyncio')
    asyncio_logger.setLevel(logging.WARNING)
    
    # Clean up expired Ask Me context caches on startup
    cleanup_expired_ask_me_caches()
    
    # Create HTTP redirect app for automatic HTTPS redirection
    redirect_app = FastAPI(title="HTTP to HTTPS Redirect")
    
    @redirect_app.get("/")
    async def redirect_root_to_https(request: Request):
        """Redirect root path to HTTPS"""
        host_header = request.headers.get("host", "localhost")
        host = host_header.split(":")[0]  # Remove port number
        https_url = f"https://{host}:8443/"
        logger.info(f"Redirecting root {request.url} to {https_url}")
        return RedirectResponse(url=https_url, status_code=301)
    
    @redirect_app.get("/{path:path}")
    async def redirect_to_https(request: Request, path: str):
        """Redirect all HTTP requests to HTTPS"""
        # Extract host without port for the redirect
        host_header = request.headers.get("host", "localhost")
        host = host_header.split(":")[0]  # Remove port number
        
        query = str(request.url.query)
        
        # Build HTTPS URL with port 8443
        https_url = f"https://{host}:8443/{path}"
        if query:
            https_url += f"?{query}"
        
        logger.info(f"Redirecting {request.url} to {https_url}")
        return RedirectResponse(url=https_url, status_code=301)
    
    def start_http_redirect():
        """Start HTTP redirect server on port 8000"""
        logger.info("Starting HTTP redirect server on port 8000...")
        config = uvicorn.Config(
            redirect_app, 
            host="0.0.0.0", 
            port=8000,
            log_level="info",
            access_log=False,  # Reduce log noise
            loop="asyncio"
        )
        server = uvicorn.Server(config)
        server.run()
    
    def start_https_main():
        """Start main HTTPS server on port 8443"""
        logger.info("Starting main HTTPS server on port 8443...")
        config = uvicorn.Config(
            app, 
            host="0.0.0.0", 
            port=8443,
            ssl_keyfile="certs/key.pem",
            ssl_certfile="certs/cert.pem",
            log_level="info",
            access_log=False,  # Reduce log noise
            loop="asyncio",
            timeout_keep_alive=30,  # Keep connections alive for 30 seconds
            timeout_graceful_shutdown=10  # Graceful shutdown timeout
        )
        server = uvicorn.Server(config)
        server.run()
    
    logger.info("🚀 Starting VeriDoc with HTTP redirect and HTTPS servers...")
    logger.info("📱 HTTP server (port 8000) will redirect to HTTPS")
    logger.info("🔒 HTTPS server (port 8443) - main application")
    logger.info("🌐 For remote access, use: https://your-ip:8443")
    
    # Start HTTP redirect server in a separate thread
    http_thread = threading.Thread(target=start_http_redirect, daemon=True)
    http_thread.start()
    
    # Start main HTTPS server in main thread
    start_https_main()
