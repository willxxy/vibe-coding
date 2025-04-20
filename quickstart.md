# Vibe Quick Start Guide

## Prerequisites

- Python 3.10 (via Conda)
- Node.js and npm
- At least 8GB RAM (16GB recommended)
- 4GB free disk space for model storage

## 5-Minute Setup

### Backend Setup

1. **Create and activate Conda environment:**
   ```bash
   conda create -n vibe python=3.10
   conda activate vibe
   ```

2. **Install backend dependencies:**
   
   Using requirements.txt:
   ```bash
   cd server
   pip install -r requirements.txt
   ```
   
   Using pyproject.toml:
   ```bash
   cd server
   pip install -e ".[dev]"
   ```

3. **Start the backend server:**
   ```bash
   python app.py
   ```
   The server will run on http://localhost:5001

### Frontend Setup

1. **Install frontend dependencies:**
   ```bash
   # From the root directory
   npm install
   ```

2. **Start the frontend development server:**
   ```bash
   npm run dev
   ```
   The application will be available at http://localhost:3000

## Usage

1. Enter text in the document editor
2. Watch as the application analyzes your text in real-time:
   - See the overall tone and subject matter analysis
   - Get context-aware responses for different parts of your document

The first analysis may take a little longer as the models are loaded into memory.

## Tips

- For optimal performance, try to keep document size reasonable (under 10,000 words)
- The system works best with well-structured text with proper sentences
- The application analyzes chunks of 3 sentences at a time by default 