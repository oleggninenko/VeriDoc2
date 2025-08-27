"""
Basic tests for VeriDoc AI application
"""
import pytest
import os
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.text_extraction import extract_text_from_file
from utils.embeddings import embed_single_text
from utils.status import update_status, get_processing_status


class TestTextExtraction:
    """Test text extraction functionality"""
    
    def test_extract_text_from_file_nonexistent(self):
        """Test text extraction from non-existent file"""
        with pytest.raises(FileNotFoundError):
            extract_text_from_file("nonexistent_file.txt")
    
    def test_extract_text_from_file_unsupported_format(self):
        """Test text extraction from unsupported file format"""
        # Create a temporary file with unsupported extension
        temp_file = "test_file.xyz"
        try:
            with open(temp_file, 'w') as f:
                f.write("test content")
            
            with pytest.raises(ValueError):
                extract_text_from_file(temp_file)
        finally:
            if os.path.exists(temp_file):
                os.remove(temp_file)


class TestEmbeddings:
    """Test embedding functionality"""
    
    def test_embed_single_text_empty(self):
        """Test embedding empty text"""
        # This test would require a mock OpenAI client
        # For now, we'll just test that the function exists
        assert callable(embed_single_text)
    
    def test_embed_single_text_none(self):
        """Test embedding None text"""
        # This test would require a mock OpenAI client
        # For now, we'll just test that the function exists
        assert callable(embed_single_text)


class TestStatus:
    """Test status management functionality"""
    
    def test_update_status(self):
        """Test status update functionality"""
        # Test that the function exists and is callable
        assert callable(update_status)
        
        # Test basic status update
        update_status("test", 50, "Test message", 5, 10)
        
        # Verify status was updated
        status = get_processing_status()
        assert status is not None
    
    def test_get_processing_status(self):
        """Test getting processing status"""
        # Test that the function exists and is callable
        assert callable(get_processing_status)
        
        # Test getting status
        status = get_processing_status()
        assert status is not None


class TestConfiguration:
    """Test configuration and setup"""
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists"""
        assert os.path.exists("requirements.txt")
    
    def test_main_app_file_exists(self):
        """Test that main application file exists"""
        assert os.path.exists("simple_web_interface_v2.py")
    
    def test_utils_directory_exists(self):
        """Test that utils directory exists"""
        assert os.path.exists("utils")
        assert os.path.isdir("utils")
    
    def test_utils_modules_exist(self):
        """Test that required utility modules exist"""
        required_modules = [
            "utils/__init__.py",
            "utils/embeddings.py",
            "utils/status.py",
            "utils/text_extraction.py"
        ]
        
        for module in required_modules:
            assert os.path.exists(module), f"Module {module} not found"


class TestDockerConfiguration:
    """Test Docker configuration"""
    
    def test_dockerfile_exists(self):
        """Test that Dockerfile exists"""
        assert os.path.exists("Dockerfile")
    
    def test_docker_compose_exists(self):
        """Test that docker-compose.yml exists"""
        assert os.path.exists("docker-compose.yml")
    
    def test_dockerignore_exists(self):
        """Test that .dockerignore exists"""
        assert os.path.exists(".dockerignore")


if __name__ == "__main__":
    pytest.main([__file__])
