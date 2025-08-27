"""
Status management utilities for processing operations.
Consolidated from duplicate implementations across the codebase.
"""

import time
import logging

logger = logging.getLogger(__name__)


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

# Global variables for deep analysis processing status
deep_analysis_status = {
    "status": "idle",
    "progress": 0,
    "current_step": "",
    "processed_items": 0,
    "total_items": 0,
    "message": "",
    "logs": []
}


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
    
    logger.info(f"Status update: {status} - {current_step or ''} - {progress or 0}%")


def update_deep_analysis_status(status: str, progress: float = None, current_step: str = None, 
                               processed_items: int = None, total_items: int = None, 
                               message: str = None, log: str = None):
    """Update deep analysis processing status"""
    global deep_analysis_status
    
    if status:
        deep_analysis_status["status"] = status
    if progress is not None:
        deep_analysis_status["progress"] = progress
    if current_step:
        deep_analysis_status["current_step"] = current_step
    if processed_items is not None:
        deep_analysis_status["processed_items"] = processed_items
    if total_items is not None:
        deep_analysis_status["total_items"] = total_items
    if message:
        deep_analysis_status["message"] = message
    if log:
        deep_analysis_status["logs"].append(f"[{time.strftime('%H:%M:%S')}] {log}")
        if len(deep_analysis_status["logs"]) > 100:
            deep_analysis_status["logs"] = deep_analysis_status["logs"][-100:]
    
    logger.info(f"Deep analysis status update: {status} - {current_step or ''} - {progress or 0}%")


def get_processing_status():
    """Get current processing status"""
    return processing_status


def get_deep_analysis_status():
    """Get current deep analysis status"""
    return deep_analysis_status


def reset_processing_status():
    """Reset processing status to idle"""
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


def reset_deep_analysis_status():
    """Reset deep analysis status to idle"""
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

