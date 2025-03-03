# RAG-CHATBOT-APP

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Flask: 2.3.3](https://img.shields.io/badge/Flask-2.3.3-lightgrey.svg)](https://flask.palletsprojects.com/)
[![Docker: Available](https://img.shields.io/badge/Docker-Available-2496ED?logo=docker&logoColor=white)](https://hub.docker.com/repository/docker/michafein/ragchatapp)
[![Test Coverage](https://img.shields.io/badge/coverage-87%25-brightgreen.svg)](https://github.com/michafein/ragchatapp/tree/main/tests)

A Retrieval-Augmented Generation (RAG) chatbot application built with Flask, leveraging LM-Studio's REST API for embeddings and model operations.  
The user interface is implemented in HTML/CSS, with assets in the `templates/` and `static/` directories. Easily deployable via Docker (here is my docker configuration: [Dockerhub](https://hub.docker.com/repository/docker/michafein/ragchatapp/general)).

---

## Table of Contents

- [Features](#features)
- [How RAG Works](#how-rag-works)
- [Flowchart](#flowchart)
- [Quick Start](#quick-start)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)
- [Testing](#testing)
- [Security Features](#security-features)
- [Development](#development)
- [Models](#models)
- [License](#license)
- [Links](#links)

---

## Features

### Core Functionality
- **Retrieval-Augmented Generation (RAG)**: Enhances LLM responses with relevant context retrieved from documents
- **PDF Document Processing**: Upload, parse, and index PDF documents for knowledge extraction
- **Vector Embeddings**: Semantic search using vector embeddings and cosine similarity to find relevant document sections

### User Experience
- **Responsive UI**: Clean and intuitive chat interface with real-time feedback
- **Source Attribution**: Transparent display of information sources with page references
- **Progress Tracking**: Real-time progress indicators for document processing

### Technical Architecture
- **Modular Design**: Well-organized codebase with clear separation of concerns
- **RESTful API**: Comprehensive endpoints for integration with other applications
- **LM-Studio Integration**: Seamless connection with local LLMs via REST API

### Development & Deployment
- **Security-Focused**: Input validation, sanitization, and secure headers protection
- **Comprehensive Testing**: Extensive test suite covering core functionalities
- **Docker Support**: Multi-stage container builds for optimized deployment
- **Detailed Logging**: Robust error handling and comprehensive logging system

---
## How RAG Works

The RAG (Retrieval-Augmented Generation) approach combines the power of retrieval-based and generation-based AI:

1. **Document Indexing**: PDF documents are processed, chunked, and converted to vector embeddings
2. **Semantic Search**: When a question is asked, the system finds the most relevant document sections
3. **Context-Enhanced Generation**: The retrieved context is sent to the language model along with the query
4. **Informed Response**: The model generates a response grounded in the specific document content

This approach improves accuracy, reduces hallucinations, and provides source attribution for responses.

## Flowchart
![Workflow Overview of the RAG Chatbot App](images/rag_chart.png)

---
## Quick Start

For developers who want to get up and running quickly:

```bash
# Clone repository
git clone https://github.com/michafein/ragchatapp.git
cd ragchatapp

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start LM-Studio and enable REST API (default port 1234)

# Run the application
python app.py

# Access the web interface
# Open http://localhost:5000 in your browser
```
**The quick start assumes:**
- You have Python 3.9+ installed
- You have LM-Studio installed with the required models
- You want to use the default configuration

For production use or custom configurations, see the detailed [Installation](#installation) and [Usage](#usage) sections.

---

## Prerequisites

- **Python 3.9+** [Install Python](https://www.python.org/downloads/)
- **LM-Studio:**[LM-Studio](https://lmstudio.ai/) Ensure LM-Studio is running as a REST API.
- **Git:** For version control.
- **Docker:** (Optional) For containerized deployment.

---

## Installation

### Option 1: Local Installation

1. **Clone the repository**:
```bash
   git clone https://github.com/michafein/ragchatapp.git
   cd ragchatapp
```

2. **Create and activate a virtual environment**:
```bash
   # On Linux/macOS
   python -m venv venv
   source venv/bin/activate
   
   # On Windows
   python -m venv venv
   venv\Scripts\activate
```

3. **Install dependencies**:
```bash
   pip install -r requirements.txt
```

4. **Set up LM-Studio**:
   - Download and install [LM-Studio](https://lmstudio.ai/)
   - Load the required models:
     - Chat Model: [`deepseek-r1-distill-qwen-7b`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)
     - Embedding Model: [`text-embedding-all-minilm-l6-v2-embedding`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
   
   **Enable REST API in LM-Studio**:
   - Go to **Developer Settings** in LM-Studio.
   - Enable the **REST API** and note the API URL (default: `http://localhost:1234`).

5. **Update `config.py`**:
   - Open the `config.py` file and ensure the following settings match your LM-Studio configuration:
     ```python
     LM_STUDIO_API_URL = "http://localhost:1234/v1/chat/completions"
     LM_STUDIO_EMBEDDING_API_URL = "http://localhost:1234/v1/embeddings"
     # Name must match LM-Studio's model
     EMBEDDING_MODEL_NAME = "text-embedding-all-minilm-l6-v2-embedding"
     CHAT_MODEL_NAME = "deepseek-r1-distill-qwen-7b"
     ```


### Option 2: Docker Deployment

1. **Build the Docker image**:
```bash
   docker build -t ragchatapp .
```

2. **Run the container**:
```bash
   docker run -p 5000:5000 \
     -v ./uploads:/app/uploads \
     -v ./embeddings:/app/embeddings \
     -v ./pages_and_chunks:/app/pages_and_chunks \
     -v ./logs:/app/logs \
     --name ragchatapp \
     ragchatapp
```

3. **Access the application**:
   - Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

4. **Alternatively, use Docker Compose**:
```bash
   docker-compose up -d
```
   **Note:** A sample `docker-compose.yml` file is provided in the repository.
---

## Usage

### 1. Start the application and access the chat interface

 **Start the Flask Server**:
   - Run the following command in your terminal:
   ```bash
     python app.py
   ```

 **Access the application**:
   - Open your browser and navigate to [http://localhost:5000](http://localhost:5000)

### 2. Upload a PDF document:
   - Click the paperclip icon
   - Select a PDF file (10MB max)
   - Wait for processing to complete (progress is shown in real-time)

### 3. Ask questions about the document:
   - Type your question in the chat input
   - The system will retrieve relevant sections from the document
   - The model will generate a response based on the retrieved content
   - Sources are provided for verification

### 4. View source information:
   - Click "Show Sources" to see where information was retrieved from
   - Each source shows the document name, page number, and relevant text


## Folder Structure

The application follows a modular, maintainable architecture:

```
ragchatapp/
├── app.py                # Main application entry point
├── routes.py             # API endpoints and route handlers
├── config.py             # Configuration settings
├── requirements.txt      # Python dependencies
├── Dockerfile            # Container configuration
├── README.md             # Project documentation
│
├── static/               # Static assets
│   └── ...               # CSS, JavaScript, images
│
├── templates/            # HTML templates
│   ├── chat.html         # Chat interface
│   └── error.html        # Error page
│
├── utils/                # Utility modules
│   ├── __init__.py       # Package initialization
│   ├── embeddings.py     # Vector embedding operations
│   ├── pdf_processing.py # PDF handling and processing
│   ├── security.py       # Security functions
│   └── text_processing.py # Text processing functions
│
├── tests/                # Test suite
│   ├── __init__.py       # Test package initialization
│   ├── test_app.py       # Tests for main application
│   ├── test_embeddings.py # Tests for embedding functions
│   ├── test_routes.py    # Tests for API routes
│   ├── test_security.py  # Tests for security functions
│   └── test_utils.py     # Tests for utility functions
│
├── logs/                 # Application logs
├── uploads/              # Directory for uploaded PDFs
├── embeddings/           # Stored vector embeddings
├── pages_and_chunks/     # Extracted text chunks from PDFs
│
├── images/               # Documentation images
└── Logo/                 # Application logos
```
---
## API Endpoints

The application provides the following REST API endpoints:

| Endpoint | Method | Description | Request Parameters | Response |
|----------|--------|-------------|-------------------|----------|
| `/` | GET | Main chat interface | None | HTML page |
| `/get` | POST | Process user messages and return responses with context-aware answers | `msg`: User message (string) | JSON: `{"response": "...", "sources": "...", "show_sources_button": true/false}` |
| `/upload` | POST | Synchronous PDF document upload and processing | `file`: PDF file (multipart/form-data) | JSON: `{"message": "...", "pdf_hash": "...", "pdf_name": "..."}` |
| `/upload_stream` | POST | Streaming PDF upload with real-time progress updates | `file`: PDF file (multipart/form-data) | Server-Sent Events with progress updates |
| `/clear_history` | POST | Clear current chat session history | None | JSON: `{"message": "Chat history cleared"}` |

### Authentication

Currently, the API doesn't require authentication as it's designed for local deployment. For production deployments, consider implementing appropriate authentication mechanisms.

### Response Formats

Most API endpoints return JSON responses with the following structure:

```json
{
  "response": "The generated response from the model",
  "sources": "Original text sources from the document (when available)",
  "show_sources_button": true,
  "status": "success"
}
```
Error responses follow this format:
```json
{
  "error": "Error message",
  "details": "Additional error details (when available)",
  "status": "error"
}
```
### Example API Usage

```python
# Example: Uploading a PDF using requests
import requests

url = "http://localhost:5000/upload"
files = {'file': open('document.pdf', 'rb')}
response = requests.post(url, files=files)
print(response.json())

# Example: Sending a chat message
message_url = "http://localhost:5000/get"
data = {'msg': 'What is in this document?'}
response = requests.post(message_url, data=data)
print(response.json()['response'])
```

---
## Troubleshooting

### Common Issues

#### LM-Studio Connection Errors

- **Problem**: "Connection refused" or "Failed to connect to LM-Studio"
  - **Solution**: Ensure LM-Studio is running and the REST API is enabled in its settings
  - **Solution**: Verify that the API URL in `config.py` matches your LM-Studio endpoint (default: `http://localhost:1234`)
  - **Solution**: Check if another application is using port 1234

- **Problem**: "Model not found" error
  - **Solution**: Verify that you have downloaded and selected the correct models in LM-Studio
  - **Solution**: Ensure model names in `config.py` match exactly with the model names in LM-Studio

#### PDF Processing Issues

- **Problem**: "Failed to process PDF" or no text extracted
  - **Solution**: Ensure the PDF contains actual text and not just scanned images
  - **Solution**: Verify the PDF is not password-protected
  - **Solution**: Try converting the PDF using an online tool to ensure compatibility

- **Problem**: "The file appears to be corrupted or not a valid PDF"
  - **Solution**: Verify the file is a valid PDF using another PDF reader
  - **Solution**: Try re-saving or re-exporting the PDF from its source

#### Memory and Performance Issues

- **Problem**: Application running slowly or crashing
  - **Solution**: Reduce the `EMBEDDING_BATCH_SIZE` in `config.py`
  - **Solution**: Process smaller PDFs or split large PDFs into smaller files
  - **Solution**: Increase available memory on your system or container

- **Problem**: Search results not relevant
  - **Solution**: Adjust the `COSINE_SIMILARITY_THRESHOLD` in `config.py` (try lower values for more results)
  - **Solution**: Review the PDF content to ensure it contains relevant information

### Viewing Logs

For more detailed troubleshooting:

- Check the application logs in the `logs/chatbot.log` file
- Look for error messages in your terminal/console where the application is running
- Enable debug mode by setting `DEBUG=True` in `config.py` for more verbose logging

### Still Having Issues?

If you're still experiencing problems:

1. Open an issue on GitHub with details about your environment and the specific error
2. Include relevant logs and error messages
3. Describe the steps to reproduce the problem

---
## Testing

Run the test suite to verify functionality:

```bash
   python -m pytest
```

For more detailed test output:

```bash
   python -m pytest -v
```
---
## Security Features

- **Input Validation**: All user inputs are validated and sanitized
- **File Validation**: Strict file type and size checks for uploads
- **Secure Headers**: Protection against XSS, clickjacking, and MIME-sniffing
- **Error Handling**: Secure error handling preventing information disclosure
- **Non-root User**: Docker container runs as a non-privileged user
---

## Development

### Adding New Features

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests to ensure functionality
5. Submit a pull request

---
## Models

The RAG Chatbot uses the following models:

- **Chat Model**: [`deepseek-r1-distill-qwen-7b`](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B) - A distilled language model from Deepseek AI
- **Embedding Model**: [`text-embedding-all-minilm-l6-v2-embedding`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) - A sentence transformer model for generating text embeddings
---
## License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

---

## Links

- [LM Studio](https://lmstudio.ai/) for the local model inference server
- [Hugging Face](https://huggingface.co/) for hosting the models
- [Flask](https://flask.palletsprojects.com/) web framework
- [PyMuPDF](https://pymupdf.readthedocs.io/) for PDF processing


