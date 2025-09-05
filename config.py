"""
Configuration settings for VeriDoc AI application.
Centralized configuration to replace hardcoded values.
"""

import os
from pathlib import Path

# Application Settings
APP_NAME = "VeriDoc AI"
APP_VERSION = "1.0.0"
APP_DESCRIPTION = "Automated Fact Verification and Document Consistency Platform"

# Server Configuration
HOST = "0.0.0.0"
PORT = 8001
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# File Upload Settings
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'.pdf', '.docx', '.xlsx', '.txt', '.xls', '.doc'}
UPLOAD_DIR = Path("uploads")

# AI Model Configuration
DEFAULT_JUDGE_MODEL = "gpt-5-mini"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-large"
DEFAULT_COMPLETION_TOKENS = 4000
DEFAULT_REASONING_EFFORT = "medium"
DEFAULT_VERBOSITY = "low"



# Text Processing Settings
CHUNK_SIZE_CHARS = 4000
OVERLAP_CHARS = 500
MAX_SNIPPET_CHARS = 4000
TOP_K_RESULTS = 10
SIMILARITY_THRESHOLD = 0.1

# Performance Settings
MAX_WORKERS = 8
BATCH_SIZE = 10
EMBEDDING_BATCH_SIZE = 100

# Cache Settings
CACHE_DIR = ".cache"
ASK_ME_CACHE_DIR = ".ask_me_cache"
CACHE_EXPIRY_HOURS = 24

# File Paths
API_CREDENTIALS_FILE = "api.txt"
CATEGORIES_FILE = Path("config/cache_categories.json")
CATEGORY_LIST_FILE = Path("config/category_list.json")
VERIFICATION_RESULTS_FILE = "verification_results.xlsx"
LOG_FILE = "veridoc.log"

# Default Categories
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

# Output Field Defaults
DEFAULT_VERIFICATION_FIELDS = [
    {"id": "par_number", "name": "Par Number", "description": "", "enabled": True},
    {"id": "par_context", "name": "Par Context", "description": "", "enabled": True},
    {"id": "is_accurate", "name": "Is Accurate", "description": "Whether the statement is accurate or not", "enabled": True},
    {"id": "degree_accuracy", "name": "Degree of Accuracy", "description": "Level of accuracy (high, medium, low)", "enabled": True},
    {"id": "inaccuracy_type", "name": "Inaccuracy Type", "description": "Type of inaccuracy if statement is inaccurate", "enabled": True},
    {"id": "description", "name": "Description", "description": "Detailed description of the verification result", "enabled": True}
]



# Role Templates
ROLE_TEMPLATES = {
    "uk_solicitor": "You are an experienced UK solicitor providing legal analysis.",
    "legal_expert": "You are a legal expert specializing in document verification.",
    "fact_checker": "You are a fact-checking specialist.",
    "ai_assistant": "You are an AI assistant helping with document analysis.",
    "senior_analyst": "You are a senior legal analyst.",
    "comprehensive_analyst": "You are a legal expert specializing in comprehensive analysis."
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "handlers": ["file", "console"]
}

# Security Settings
CORS_ORIGINS = ["*"]  # Configure appropriately for production
ALLOW_CREDENTIALS = True
ALLOWED_METHODS = ["*"]
ALLOWED_HEADERS = ["*"]

# Timeout Settings
REQUEST_TIMEOUT = 30
UPLOAD_TIMEOUT = 300
PROCESSING_TIMEOUT = 3600

# Rate Limiting
RATE_LIMIT_REQUESTS = 100
RATE_LIMIT_WINDOW = 60  # seconds

# Environment-specific overrides
if os.getenv("ENVIRONMENT") == "production":
    DEBUG = False
    CORS_ORIGINS = ["https://yourdomain.com"]  # Configure for production
    LOGGING_CONFIG["level"] = "WARNING"
elif os.getenv("ENVIRONMENT") == "development":
    DEBUG = True
    LOGGING_CONFIG["level"] = "DEBUG"
