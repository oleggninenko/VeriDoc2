# Document Verification System - Simple Web Interface

A lightweight web interface for the AI-powered document verification system that works without requiring Node.js installation.

## 🚀 Features

- **No Node.js Required**: Pure Python solution with embedded HTML/CSS/JavaScript
- **File Upload**: Drag-and-drop file upload with support for TXT, PDF, DOCX, and XLSX
- **Real-time Processing**: Live progress tracking with status updates
- **Results Viewing**: Interactive results display with download functionality
- **Modern UI**: Clean, responsive interface built with vanilla HTML/CSS/JavaScript

## 📁 Files

- `simple_web_interface.py` - Main web interface application
- `start_web_interface.bat` - Windows startup script
- `verify_statements.py` - Original verification script
- `api.txt` - API credentials file

## 🛠️ Setup Instructions

### Prerequisites

- **Python** (v3.8 or higher)
- **pip** (Python package manager)

### 1. Install Dependencies

```bash
pip install fastapi uvicorn websockets python-multipart openpyxl openai numpy tenacity tqdm scikit-learn PyPDF2 python-docx pandas
```

### 2. Configure API Credentials

Ensure your `api.txt` file is in the root directory with your OpenAI API credentials:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com
```

## 🚀 Running the Application

### Option 1: Using Batch File (Windows)

```bash
start_web_interface.bat
```

### Option 2: Manual Startup

```bash
python simple_web_interface.py
```

### Option 3: Direct Python Execution

```bash
python -m uvicorn simple_web_interface:app --host 0.0.0.0 --port 8000
```

## 🌐 Using the Web Interface

### 1. Access the Interface

1. Start the web interface using one of the methods above
2. Open your web browser
3. Navigate to `http://localhost:8000`
4. You'll see the Document Verification System interface

### 2. Upload Files

1. **Click the "Upload Files" tab** (default)
2. **Click the upload area** or drag and drop files
3. **Select your files**:
   - **Source files**: Your large text documents (TXT, PDF, DOCX, XLSX)
   - **Statements file**: Excel file with "Par" and "Content" columns
4. **Review uploaded files** in the file list
5. **Click "Start Verification Process"** when ready

### 3. Monitor Processing

1. **Switch to the "Processing" tab**
2. **Watch real-time progress**:
   - Current step being processed
   - Progress bar showing completion percentage
   - Number of items processed vs total
   - Status messages and logs
3. **Wait for completion** - the system will automatically update

### 4. View Results

1. **Switch to the "Results" tab**
2. **Click "Load Results"** to display verification results
3. **Review the results table** showing:
   - Paragraph numbers
   - Context information
   - Accuracy verdicts
   - Degree of accuracy scores
   - Inaccuracy types and descriptions
4. **Click "Download Results"** to save as Excel file

## 🔧 API Endpoints

The web interface provides these REST API endpoints:

- `GET /` - Main web interface
- `POST /api/upload` - Upload files
- `POST /api/process` - Start verification process
- `GET /api/status` - Get processing status
- `GET /api/results` - Get verification results
- `GET /api/results/download` - Download results as Excel
- `GET /api/caches` - Get list of cached files
- `DELETE /api/caches` - Delete selected caches

## 🎨 Interface Features

### Upload Tab
- **Drag & Drop**: Simply drag files onto the upload area
- **File Type Detection**: Automatically detects source vs statements files
- **File Management**: Remove individual files or see file sizes
- **Validation**: Ensures you have both source and statements files

### Processing Tab
- **Real-time Updates**: Progress updates every second
- **Visual Progress Bar**: Shows completion percentage
- **Status Information**: Current step, processed items, messages
- **Processing Logs**: Detailed logs of what's happening

### Results Tab
- **Interactive Table**: Sortable and searchable results
- **Statistics**: Overview of accuracy rates
- **Download Function**: Export results to Excel
- **Filter Options**: Filter by accuracy status

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**:
   ```
   Error: [Errno 10048] Only one usage of each socket address is normally permitted
   ```
   **Solution**: Change the port in `simple_web_interface.py` (line 500) or kill the process using port 8000

2. **API Connection Issues**:
   ```
   Error: OPENAI_API_KEY not found in api.txt
   ```
   **Solution**: Ensure `api.txt` exists in the root directory with correct API credentials

3. **File Upload Issues**:
   ```
   Error: Upload failed
   ```
   **Solution**: Check file permissions and ensure supported file formats

4. **Processing Errors**:
   ```
   Error: Processing failed
   ```
   **Solution**: Check the logs in the Processing tab for detailed error messages

### Debug Mode

To run with detailed logging:

```bash
python -m uvicorn simple_web_interface:app --host 0.0.0.0 --port 8000 --log-level debug
```

## 📊 Performance

- **File Processing**: Supports large files with progress tracking
- **Caching**: Automatic caching of processed documents
- **Parallel Processing**: Multiple statements processed concurrently
- **Memory Efficient**: Streams large files without loading entirely into memory

## 🔒 Security

- **Local Only**: Runs on localhost by default
- **File Validation**: Validates file types before processing
- **API Security**: Credentials stored locally in `api.txt`
- **No External Dependencies**: All processing happens locally

## 🚀 Advanced Usage

### Custom Port

To run on a different port:

```bash
python -m uvicorn simple_web_interface:app --host 0.0.0.0 --port 8080
```

### Network Access

To allow access from other computers on the network:

```bash
python -m uvicorn simple_web_interface:app --host 0.0.0.0 --port 8000
```

### Production Deployment

For production use:

```bash
pip install gunicorn
gunicorn simple_web_interface:app -w 4 -k uvicorn.workers.UvicornWorker
```

## 📝 Integration with Original Script

This web interface is a wrapper around your existing `verify_statements.py` script:

- **Same Logic**: Uses all the same verification algorithms
- **Same Output**: Produces identical Excel results
- **Same Caching**: Uses the same cache system
- **Same Configuration**: Uses the same `api.txt` and settings

## 🤝 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the console output for error messages
3. Check the Processing tab logs for detailed information
4. Ensure all dependencies are installed correctly

## 📞 Quick Start Checklist

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install fastapi uvicorn...`)
- [ ] `api.txt` file configured with API credentials
- [ ] `verify_statements.py` script working
- [ ] Run `python simple_web_interface.py`
- [ ] Open browser to `http://localhost:8000`
- [ ] Upload source and statements files
- [ ] Start verification process
- [ ] Monitor progress and view results

## 🎉 Success!

Once everything is working, you'll have a modern web interface for your document verification system that's easy to use and provides real-time feedback on the verification process!
