# Vibe Architecture

## Data Flow

```
┌───────────────┐     ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│               │     │               │     │               │     │               │
│   Document    │────►│  Global LLM   │────►│   Local LLM   │────►│   Response    │
│     Text      │     │   Analysis    │     │    Analysis   │     │    Stream     │
│               │     │               │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘     └───────────────┘
                            │                      ▲
                            │                      │
                            ▼                      │
                      ┌───────────────┐           │
                      │               │           │
                      │    Context    │───────────┘
                      │  Information  │
                      │               │
                      └───────────────┘
```

## Processing Flow

1. **Document Input**:
   - The user types or pastes document text into the frontend editor
   - The text is sent to the backend for analysis

2. **Global LLM Analysis**:
   - The entire document is analyzed by the Global LLM
   - The Global LLM identifies:
     - Tone (e.g., formal, casual, technical)
     - Subject matter (e.g., journal entry, academic paper)
     - Overall context summary
   - The analysis is streamed to the frontend as a `global_analysis` event

3. **Document Chunking**:
   - The backend splits the document into manageable chunks (typically 3 sentences each)
   - Each chunk is processed separately by the Local LLM

4. **Local LLM Analysis**:
   - Each chunk is analyzed in the context of the global analysis
   - The Local LLM understands both the local content and the global context
   - It generates appropriate responses tailored to each chunk
   - The analysis is streamed token by token to the frontend

5. **Response Display**:
   - The frontend receives and displays the streaming responses
   - The UI shows both the global document analysis and chunk-specific responses

## Component Architecture

### Backend Components

1. **Flask Server**:
   - Handles HTTP requests and SSE streaming
   - Manages the analysis workflow

2. **Text Processing**:
   - Splits documents into sentences and chunks
   - Prepares text for LLM analysis

3. **LLM Integration**:
   - Loads and manages Gemma 3 models
   - Handles inference and token generation
   - Implements streaming generation

### Frontend Components

1. **Document Editor**:
   - Captures and displays user text input
   - Handles text selection and cursor positioning

2. **Analysis Display**:
   - Shows global document analysis (tone, subject, context)
   - Displays chunk-specific analyses

3. **Streaming Client**:
   - Manages SSE connection to the backend
   - Handles different event types (global_analysis, chunk, token)
   - Updates UI components in real-time

## Model Prompt Architecture

### Global LLM Prompts

The Global LLM receives three separate prompts:

1. **Tone Analysis**: 
   ```
   Analyze the following text and determine its primary purpose or tone.
   Respond with only ONE OR TWO words from this list: 
   [Personal Reflection, Academic Note, Creative Writing, Technical Instruction, 
   Correction Needed, Idea Generation, Factual Summary, Question Posing].
   ```

2. **Subject Matter Analysis**:
   ```
   Analyze the following text and determine its subject matter or domain.
   Respond with only ONE OR TWO words describing what this text is about.
   ```

3. **Context Summary**:
   ```
   Provide a brief context summary of the following text.
   ```

### Local LLM Prompts

The Local LLM receives a system message containing the global context:

```
You are analyzing text that has been identified as:
- Tone: [determined tone]
- Subject: [subject matter]
- Context: [context summary]

Provide a response appropriate for this context.
```

And a user message with the specific chunk:

```
Given the overall document context, analyze the following excerpt and provide a helpful response.
``` 