# Document Verification System - Web Interface

A modern web interface for the AI-powered document verification system, built with React, Next.js, and Tailwind CSS.

## 🚀 Features

- **Modern UI**: Clean, responsive interface built with React and Tailwind CSS
- **File Upload**: Drag-and-drop file upload with support for TXT, PDF, DOCX, and XLSX
- **Cache Management**: Visual interface for managing cached source files
- **Real-time Processing**: Live progress tracking with WebSocket updates
- **Results Analysis**: Interactive results viewer with filtering and search
- **Download Results**: Export verification results as Excel files

## 📁 Project Structure

```
├── frontend/                 # Next.js React frontend
│   ├── app/                 # Next.js app directory
│   ├── components/          # React components
│   ├── hooks/              # Custom React hooks
│   └── package.json        # Frontend dependencies
├── backend/                 # FastAPI Python backend
│   ├── main.py             # FastAPI application
│   └── requirements.txt    # Python dependencies
├── verify_statements.py    # Original verification script
├── start_frontend.bat      # Frontend startup script
├── start_backend.bat       # Backend startup script
└── README_WEB.md          # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- **Node.js** (v18 or higher)
- **Python** (v3.8 or higher)
- **npm** or **yarn**

### 1. Install Frontend Dependencies

```bash
cd frontend
npm install
```

### 2. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 3. Configure API Credentials

Ensure your `api.txt` file is in the root directory with your OpenAI API credentials:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_BASE_URL=https://api.openai.com
```

## 🚀 Running the Application

### Option 1: Using Batch Files (Windows)

1. **Start the Backend**:
   ```bash
   start_backend.bat
   ```
   This will start the FastAPI server on `http://localhost:8000`

2. **Start the Frontend**:
   ```bash
   start_frontend.bat
   ```
   This will start the Next.js development server on `http://localhost:3000`

### Option 2: Manual Startup

1. **Start Backend**:
   ```bash
   cd backend
   python main.py
   ```

2. **Start Frontend** (in a new terminal):
   ```bash
   cd frontend
   npm run dev
   ```

## 🌐 Using the Web Interface

### 1. Upload Files
- Navigate to the "Upload Files" tab
- Drag and drop your source documents and statements files
- Supported formats: TXT, PDF, DOCX, XLSX
- Click "Start Verification Process" to begin

### 2. Manage Caches
- Use the "Cache Manager" tab to view and manage cached source files
- Select files to delete or view cache sizes
- Caches are automatically created when processing source files

### 3. Monitor Processing
- Switch to the "Processing" tab to see real-time progress
- View detailed logs and current processing step
- Progress updates automatically via WebSocket

### 4. View Results
- Check the "Results" tab to see verification results
- Filter by accuracy status
- Search through results
- Download results as Excel file

## 🔧 API Endpoints

The backend provides the following REST API endpoints:

- `POST /api/upload` - Upload files
- `POST /api/process` - Start verification process
- `GET /api/caches` - Get list of cached files
- `DELETE /api/caches` - Delete selected caches
- `GET /api/results` - Get verification results
- `GET /api/results/download` - Download results as Excel
- `GET /api/status` - Get processing status

## 🔌 WebSocket Events

Real-time updates are provided via WebSocket on `ws://localhost:8001`:

- `processing_update` - Processing status updates

## 🎨 Customization

### Styling
The interface uses Tailwind CSS. You can customize the design by modifying:
- `frontend/tailwind.config.ts` - Tailwind configuration
- `frontend/app/globals.css` - Global styles

### Components
All React components are in `frontend/components/` and can be customized as needed.

## 🐛 Troubleshooting

### Common Issues

1. **Port Already in Use**:
   - Backend: Change port in `backend/main.py` (line 200)
   - Frontend: Change port in `frontend/next.config.js`

2. **API Connection Issues**:
   - Ensure `api.txt` is in the root directory
   - Check that the backend is running on port 8000
   - Verify CORS settings in `backend/main.py`

3. **File Upload Issues**:
   - Check file permissions in the `backend/uploads/` directory
   - Ensure supported file formats are being uploaded

### Debug Mode

To run in debug mode:

**Backend**:
```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Frontend**:
```bash
cd frontend
npm run dev -- --debug
```

## 📊 Performance Optimization

- **Batch Processing**: Files are processed in batches for better performance
- **Caching**: Source files are cached to avoid reprocessing
- **Parallel Processing**: Multiple statements are processed concurrently
- **Progress Tracking**: Real-time updates prevent timeout issues

## 🔒 Security Considerations

- API credentials are stored locally in `api.txt`
- File uploads are validated for supported formats
- CORS is configured for local development only
- WebSocket connections are managed securely

## 🚀 Deployment

### Production Setup

1. **Build Frontend**:
   ```bash
   cd frontend
   npm run build
   npm start
   ```

2. **Deploy Backend**:
   ```bash
   cd backend
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

3. **Environment Variables**:
   - Set `OPENAI_API_KEY` and `OPENAI_BASE_URL` as environment variables
   - Configure CORS origins for production domain
   - Set up proper SSL certificates

## 📝 License

This project is part of the Document Verification System. See the main project license for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation
3. Check the original script documentation
4. Create an issue in the repository
