# Vibe - Context-Aware Document Assistant

**NOTE: THIS PROJECT IS ALMOST 99% AI GENERATED. THIS IS JUST AN EXERCISE AND SEEING HOW WELL WE CAN USE LLMS TO CREATE A PRODUCT.**

Vibe is an intelligent document assistant that provides context-aware responses as you write. The application uses two LLMs (Large Language Models) to analyze your document:

1. **Global LLM**: Analyzes the entire document to determine tone, subject matter, and overall context
2. **Local LLM**: Receives context from the Global LLM and provides appropriate responses for specific sections of text

## Architecture

- **Frontend**: React/Next.js web application
- **Backend**: Python Flask server serving LLM models

### System Architecture
```
┌─────────────┐         ┌────────────────┐         ┌─────────────────┐
│             │  HTTP/  │                │         │                 │
│  Frontend   │  SSE    │  Flask Backend │         │  LLM Models     │
│  (Next.js)  ◄─────────►  (Python)      ◄─────────►  (Gemma 3)      │
│             │         │                │         │                 │
└─────────────┘         └────────────────┘         └─────────────────┘
```

## Requirements

### Frontend
- Node.js (14.x or higher)
- npm/yarn

### Backend
- Python 3.10
- Conda (for environment management)

## Setup Instructions

### Backend Setup

1. Create and activate the conda environment:
   ```bash
   conda create -n vibe python=3.10
   conda activate vibe
   ```

2. Install dependencies:
   
   **Using requirements.txt**:
   ```bash
   cd server
   pip install -r requirements.txt
   ```

   **Using pyproject.toml**:
   ```bash
   cd server
   pip install -e ".[dev]"
   ```

3. If no dependency files are available, install the following packages:
   ```bash
   pip install flask flask-cors torch transformers accelerate numpy
   ```
   
4. For Mac users with Apple Silicon, ensure PyTorch is installed with MPS support:
   ```bash
   pip install torch torchvision
   ```

### Frontend Setup

1. Navigate to the root directory and install dependencies:
   ```bash
   npm install
   ```

## Running the Application

### Start the Backend Server

1. Activate the conda environment:
   ```bash
   conda activate vibe
   ```

2. Navigate to the server directory and run the Flask application:
   ```bash
   cd server
   python app.py
   ```
   The backend server will start on http://localhost:5001

### Start the Frontend Development Server

1. From the root directory, run:
   ```bash
   npm run dev
   ```
   The frontend will be available at http://localhost:3000

## How It Works

1. As you write in the document editor, the content is sent to the backend server
2. The Global LLM analyzes the entire document to determine:
   - Tone (e.g., serious, happy, objective)
   - Subject matter (e.g., journal entry, school notes)
   - Overall context summary
3. This information is passed to the Local LLM, which analyzes specific chunks of text
4. The Local LLM generates responses appropriate to the context, which are streamed back to the frontend
5. The frontend displays these context-aware responses as you write

## Frontend Implementation Guide

The frontend should implement:

1. **Document Editor**: A text editor component that captures user input
2. **Analysis Display**: Components to display the global analysis and chunk-specific responses
3. **SSE Client**: Logic to connect to the backend's Server-Sent Events stream

Example of connecting to the backend's SSE stream:

```typescript
function connectToAnalysisStream(documentText: string) {
  // Close any existing connection
  if (eventSource) {
    eventSource.close();
  }
  
  // Set up a new SSE connection
  const url = new URL('/analyze', API_BASE_URL);
  const eventSource = new EventSource(url);
  
  // Handle different event types
  eventSource.addEventListener('global_analysis', (event) => {
    const globalAnalysis = JSON.parse(event.data);
    // Update UI with tone, subject matter, and context summary
  });
  
  eventSource.addEventListener('chunk', (event) => {
    const chunkData = JSON.parse(event.data);
    // Update UI with chunk analysis
  });
  
  eventSource.addEventListener('token', (event) => {
    const tokenData = JSON.parse(event.data);
    // Update UI with streaming tokens
  });
  
  eventSource.addEventListener('error', (event) => {
    console.error('Error from SSE stream:', event.data);
    eventSource.close();
  });
  
  eventSource.addEventListener('end', () => {
    eventSource.close();
  });
  
  // Send the document text to be analyzed
  fetch('/analyze', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      text: documentText,
      chunk_size: 3, // Adjust as needed
    }),
  });
  
  return eventSource;
}
```

## API Endpoints

### POST /analyze

Analyzes document text and returns context-aware responses.

**Request Body:**
```json
{
  "text": "Your document text here",
  "chunk_size": 3  // Optional: Number of sentences per chunk
}
```

**Response:**
The endpoint returns a Server-Sent Events (SSE) stream with the following event types:
- `global_analysis`: Initial document-wide analysis
- `chunk`: Analysis for a specific text chunk
- `token`: Individual tokens for streaming responses
- `chunk_complete`: Signals completion of a chunk analysis
- `end`: Signals the end of the stream
- `error`: Contains error information if something goes wrong

## Model Information

The application uses Google's Gemma 3 models:
- Model ID: `google/gemma-3-1b-it`
- The same model is used for both global and local analysis, but with different prompts

## Troubleshooting

### Common Issues

1. **Models not loading**: Ensure you have sufficient disk space and RAM. The models require approximately 2GB of disk space and at least 8GB of RAM.

2. **CUDA/MPS not available**: If you're not seeing GPU acceleration:
   - For NVIDIA GPUs: Ensure CUDA drivers are installed correctly
   - For Apple Silicon: Ensure you have the latest PyTorch version with MPS support

3. **Backend connection issues**: Verify that the backend server is running on port 5001 and that there are no firewall restrictions.

4. **Frontend not connecting to backend**: Check CORS settings in the Flask app and ensure the frontend is using the correct API URL.
