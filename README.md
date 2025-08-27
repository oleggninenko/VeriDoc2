# VeriDoc AI - Automated Fact Verification and Document Consistency Platform

A comprehensive web-based platform for automated fact verification and document consistency analysis using AI-powered semantic search and verification.

## 🚀 Features

### Core Functionality
- **Document Verification**: Verify statements against a knowledge database using AI
- **Deep Analysis**: Comprehensive legal analysis with appeal grounds assessment
- **Interactive Q&A**: Ask questions about your knowledge database with conversation history
- **Document Management**: Upload and categorize various document types (PDF, Word, Excel, Text)
- **Caching System**: Efficient document indexing and caching for improved performance

### Document Support
- **Knowledge Database**: PDF, Word (DOCX, DOC), Excel (XLSX, XLS), Text (TXT)
- **Statements File**: Excel files with 'Par' and 'Content' columns
- **Context Files**: Additional documents for conversation context

### AI Models
- **Judge Model**: GPT-5-mini for statement verification
- **Embedding Model**: text-embedding-3-large for semantic search
- **Deep Analysis**: GPT-5-mini for comprehensive legal analysis

## 📋 Prerequisites

- Python 3.8 or higher
- OpenAI API key (or compatible API endpoint)
- Required Python packages (see `requirements.txt`)

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/veridoc-ai.git
   cd veridoc-ai
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API credentials**
   Create an `api.txt` file in the project root with your API configuration:
   ```
   api_key=your_openai_api_key_here
   base_url=https://api.openai.com/v1
   ```

## 🚀 Usage

### Starting the Application

1. **Run the web interface**
   ```bash
   python simple_web_interface_v2.py
   ```

2. **Access the application**
   Open your browser and navigate to `http://localhost:8000`

### Basic Workflow

1. **Upload Knowledge Files**
   - Go to the "Knowledge Database" section
   - Upload your source documents (PDF, Word, Excel, Text)
   - Click "Index & Cache Files" to process them

2. **Upload Statements for Verification**
   - Go to the "Statements for Verification" section
   - Upload an Excel file with 'Par' and 'Content' columns
   - Select which statements to verify

3. **Run Verification**
   - Click "Verification Process" for standard verification
   - Click "Deep Analysis" for comprehensive legal analysis

4. **Ask Questions**
   - Use the "Ask Me" section to query your knowledge database
   - Upload additional context files for more specific answers

## 📁 Project Structure

```
veridoc-ai/
├── simple_web_interface_v2.py    # Main web application
├── verify_statements.py          # Core verification logic
├── requirements.txt              # Python dependencies
├── api.txt                       # API configuration (not in repo)
├── utils/                        # Utility modules
│   ├── embeddings.py            # Embedding utilities
│   ├── text_extraction.py       # Document text extraction
│   └── status.py                # Status management
├── uploads/                      # Uploaded files (not in repo)
├── .index_cache/                 # Document cache (not in repo)
├── .ask_me_cache/               # Conversation cache (not in repo)
└── legacy/                       # Legacy files
```

## ⚙️ Configuration

### API Configuration
The application uses OpenAI-compatible APIs. Configure your API settings in `api.txt`:
```
api_key=your_api_key_here
base_url=https://api.openai.com/v1
```

### Technical Parameters
- **Chunk Size**: 4,000 characters with 500-character overlap
- **Top-K Results**: 10 most relevant chunks per query
- **Parallel Workers**: 8 concurrent threads
- **Batch Size**: 10 statements per batch
- **Max Tokens**: 4,000 completion tokens

## 🔧 Features in Detail

### Document Categorization
- Automatic categorization based on document names
- Manual category assignment
- Bulk category operations
- Custom category creation

### Verification Process
1. **Document Chunking**: Source documents split into 4,000-character chunks
2. **Vector Embedding**: Chunks converted to high-dimensional vectors
3. **Similarity Search**: Find most relevant chunks for each statement
4. **AI Verification**: GPT-5-mini analyzes statements against evidence
5. **Results Export**: Excel output with detailed analysis

### Deep Analysis
- Comprehensive legal analysis
- Appeal grounds assessment
- High reasoning effort analysis
- Detailed explanations and recommendations

### Interactive Q&A
- Conversation history tracking
- Context file uploads
- Follow-up question support
- Confidence scoring
- Key points extraction

## 🔒 Security Considerations

- API keys are stored locally in `api.txt` (not committed to repository)
- Uploaded files are stored in local `uploads/` directory
- Cache files contain processed document data
- No sensitive data is transmitted to external services beyond API calls

## 🐛 Troubleshooting

### Common Issues

1. **API Key Issues**
   - Ensure `api.txt` exists and contains valid API credentials
   - Check API key permissions and quota

2. **File Upload Problems**
   - Verify file formats are supported
   - Check file size limits
   - Ensure proper file permissions

3. **Performance Issues**
   - Reduce parallel workers if system is overloaded
   - Clear cache directories if needed
   - Monitor memory usage with large documents

### Logs
Check `veridoc.log` for detailed error messages and debugging information.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing the AI models and API
- FastAPI for the web framework
- The open-source community for various supporting libraries

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section above
- Review the logs in `veridoc.log`

---

**Note**: This application is designed for document verification and legal analysis. Always review AI-generated results and use them as a tool to assist human judgment, not as a replacement for professional legal advice.
