import torch
from flask import Flask, request, jsonify, Response # Import Response
from flask_cors import CORS # Import CORS
from transformers import AutoTokenizer, Gemma3ForCausalLM
import logging
import re # For simple sentence splitting
import json # Import json for formatting SSE data
import functools
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app) # Enable CORS for the entire app

# --- Device Setup ---
def get_device():
    """Get the best available compute device."""
    try:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    except Exception as e:
        logging.warning(f"Error detecting device, defaulting to CPU: {e}")
        return torch.device("cpu")

DEVICE = get_device()
logging.info(f"Using device: {DEVICE}")
# --- End Device Setup ---

# --- Model Loading ---
MODEL_ID = "google/gemma-3-1b-it"  # Using the same model for both global and local LLM

# Global variables for models and tokenizers
model = None
tokenizer = None

# Add model loading lock to prevent concurrent loading
model_loading_lock = threading.Lock()

def load_model_and_tokenizer():
    """Load the LLM model and tokenizer."""
    global model, tokenizer
    
    with model_loading_lock:
        # Skip loading if model is already loaded
        if model is not None and tokenizer is not None:
            return True
        
        # Load Model
        logging.info(f"Loading model: {MODEL_ID}...")
        try:
            start_time = time.time()
            model = Gemma3ForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            logging.info(f"Model loaded successfully onto {DEVICE} in {time.time() - start_time:.2f}s.")
            return True
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            return False

# Add timeout wrapper for inference
def timeout(seconds):
    """Decorator to add a timeout to a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            result = [None]
            error = [None]
            
            def target():
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    error[0] = e
            
            thread = threading.Thread(target=target)
            thread.daemon = True
            thread.start()
            thread.join(seconds)
            
            if thread.is_alive():
                raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
            if error[0]:
                raise error[0]
            return result[0]
        return wrapper
    return decorator

# Attempt to load model at startup
try:
    models_ready = load_model_and_tokenizer()
    if not models_ready:
        logging.warning("Failed to load model on startup. Will retry on first request.")
except Exception as e:
    logging.error(f"Error during startup model loading: {e}")
    models_ready = False
# --- End Model Loading ---

# --- Inference Helper ---
@timeout(60)
def run_inference(prompt_text, system_message="You are a helpful assistant.", max_new_tokens=128, stream=False):
    """Runs inference using the model and tokenizer."""
    
    if not model or not tokenizer:
        logging.error("Model or tokenizer not loaded, cannot run inference.")
        return None

    # Truncate very long inputs to prevent slow processing
    max_input_length = 1024
    if len(prompt_text) > max_input_length:
        logging.warning(f"Input text truncated from {len(prompt_text)} to {max_input_length} chars")
        prompt_text = prompt_text[:max_input_length] + "..."

    messages = [[
        {
            "role": "system",
            "content": [{"type": "text", "text": system_message}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        },
    ]]

    try:
        # Use the tokenizer's chat template and move inputs to the correct device
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE)

        input_token_len = inputs["input_ids"].shape[1]
        logging.info(f"Running inference with {input_token_len} input tokens and max_new_tokens={max_new_tokens}")
        
        if input_token_len > 2048:
            logging.warning(f"Very large input: {input_token_len} tokens, performance may be affected")
        
        if stream:
            # For streaming mode, use the streaming generator
            return generate_stream(inputs, input_token_len, max_new_tokens)
        else:
            # Non-streaming mode (batch generation)
            start_time = time.time()
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens, 
                    do_sample=False,
                    # Clear unneeded params
                    top_p=None,
                    top_k=None
                )

            # Decode the output tokens
            generated_tokens = outputs[0][input_token_len:]
            result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            elapsed = time.time() - start_time
            logging.info(f"Inference completed successfully in {elapsed:.2f}s")
            return result_text.strip()

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return None

@timeout(30)
def generate_stream(inputs, input_token_len, max_new_tokens=128):
    """Generator function that yields tokens as they're generated."""
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer not provided for streaming.")
        return
        
    with torch.inference_mode():
        generation = inputs["input_ids"].clone()
        
        for _ in range(max_new_tokens):
            # Run model on current sequence
            outputs = model(generation)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Get the next token (greedy selection)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Break if EOS token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Append token to the sequence
            generation = torch.cat([generation, next_token], dim=-1)
            
            # Decode just the new token
            new_token_text = tokenizer.decode(next_token[0], skip_special_tokens=True)
            
            # Only yield if there's actual text
            if new_token_text:
                yield new_token_text
                # Small delay for visible streaming (can be adjusted based on needs)
                time.sleep(0.01)

# --- End Inference Helper ---

# --- Text Processing Helpers ---
def split_into_sentences(text):
    """Split text into sentences using regex pattern matching."""
    if not text or not text.strip():
        return []
        
    # Clean up text - normalize spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add space after punctuation for splitting, handle existing spaces
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    
    # Split by punctuation followed by space
    parts = re.split(r'(?<=[.!?])\s+', text)
    
    # Filter empty strings and strip whitespace
    sentences = [part.strip() for part in parts if part.strip()]
    
    # Handle case where splitting didn't work (e.g., no punctuation)
    if not sentences and text.strip():
        sentences = [text.strip()]
    
    return sentences

def create_text_chunks(text, chunk_size_n=3, overlap=1):
    """Split text into overlapping chunks of roughly n sentences each."""
    sentences = split_into_sentences(text)
    
    # If no sentences found, return the whole text as one chunk
    if not sentences:
        return [text] if text.strip() else []
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        # Get chunk of size chunk_size_n or remaining sentences
        end_idx = min(i + chunk_size_n, len(sentences))
        chunk = " ".join(sentences[i:end_idx])
        chunks.append(chunk)
        
        # Move window with overlap
        i = i + max(1, chunk_size_n - overlap)
    
    return chunks

# --- End Text Processing Helpers ---


@app.route('/')
def home():
    return "Backend server is running!"

# Main analysis endpoint
@app.route('/analyze', methods=['POST'])
def analyze_document():
    global models_ready
    
    # Try to load models if they aren't ready yet
    if not models_ready:
        logging.info("Model not loaded yet, attempting to load now...")
        models_ready = load_model_and_tokenizer()
    
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded. Check server logs or try again later."}), 503

    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' in request body"}), 400
        
    chunk_size_n = data.get('chunk_size', 3)
    if not isinstance(chunk_size_n, int) or chunk_size_n <= 0:
        return jsonify({"error": "'chunk_size' must be a positive integer"}), 400
        
    full_text = data['text']
    if not full_text.strip():
        return jsonify({"error": "'text' cannot be empty"}), 400

    def generate_analysis():
        try:
            # --- Global LLM Analysis (Document-wide) ---
            start_time = time.time()
            logging.info("Performing global document analysis...")
            
            # Send initial progress update to client
            progress_data = {"message": "Starting document analysis..."}
            progress_payload = json.dumps(progress_data)
            yield f"event: progress\ndata: {progress_payload}\n\n"
            
            # Limit text size to improve performance
            text_for_analysis = full_text
            if len(text_for_analysis) > 8000:
                logging.warning(f"Document too large ({len(text_for_analysis)} chars), truncating for analysis")
                text_for_analysis = full_text[:8000] + "... [document truncated for analysis]"
            
            # More comprehensive global analysis in a single prompt with shorter expected response
            global_prompt = f"""Analyze the following text and provide these specific details:
1. PRIMARY TONE: ONE word describing tone (e.g., casual, formal, academic)
2. SUBJECT MATTER: ONE or TWO words describing topic
3. CONTEXT SUMMARY: 1-2 sentences summarizing the text

Be very concise. Keep your response short.

Text:
---
{text_for_analysis}
---

Format your response exactly like this:
TONE: [one word]
SUBJECT: [one or two words]
SUMMARY: [1-2 sentence summary]"""
            
            # Progress update
            progress_data = {"message": "Analyzing document context..."}
            progress_payload = json.dumps(progress_data)
            yield f"event: progress\ndata: {progress_payload}\n\n"
            
            global_analysis_text = run_inference(
                global_prompt, 
                system_message="You are an expert text analyst. Be concise.", 
                max_new_tokens=100
            )

            if not global_analysis_text:
                logging.error("Failed to complete global analysis. Stopping stream.")
                error_payload = json.dumps({"error": "Failed to determine global text attributes."})
                yield f"event: error\ndata: {error_payload}\n\n"
                return
            
            first_stage_time = time.time() - start_time
            logging.info(f"Global analysis completed in {first_stage_time:.2f}s")
            
            # Parse the structured response
            tone_match = re.search(r'TONE:\s*(.*?)(?:\n|$)', global_analysis_text)
            subject_match = re.search(r'SUBJECT:\s*(.*?)(?:\n|$)', global_analysis_text)
            summary_match = re.search(r'SUMMARY:\s*(.*?)(?:\n|$)', global_analysis_text, re.DOTALL)
            
            tone = tone_match.group(1).strip() if tone_match else "Unknown"
            subject_matter = subject_match.group(1).strip() if subject_match else "Unknown"
            context_summary = summary_match.group(1).strip() if summary_match else "No summary available"
            
            # Combine global analysis results
            global_analysis = {
                "tone": tone,
                "subject_matter": subject_matter,
                "context_summary": context_summary
            }
            
            # Send global analysis to client
            global_payload = json.dumps(global_analysis)
            yield f"event: global_analysis\ndata: {global_payload}\n\n"
            
            # --- Local LLM Processing (Chunk-based) ---
            logging.info(f"Chunking text for local processing...")
            
            # Progress update
            progress_data = {"message": "Preparing text chunks..."}
            progress_payload = json.dumps(progress_data)
            yield f"event: progress\ndata: {progress_payload}\n\n"
            
            # Use smaller chunk size for faster processing
            effective_chunk_size = min(chunk_size_n, 2)
            text_chunks = create_text_chunks(full_text, effective_chunk_size, overlap=0)
            
            # Limit number of chunks for performance
            max_chunks = 10
            if len(text_chunks) > max_chunks:
                logging.warning(f"Too many chunks ({len(text_chunks)}), limiting to {max_chunks}")
                text_chunks = text_chunks[:max_chunks]
            
            # Handle empty text case
            if not text_chunks:
                text_chunks = [""]
                
            logging.info(f"Created {len(text_chunks)} chunks for analysis")

            # Create a more concise system message
            system_message_local = f"""You are analyzing a document with tone: {tone}, subject: {subject_matter}.
Provide brief, helpful responses about each excerpt."""

            # Process chunks in sequence
            for chunk_index, chunk_content in enumerate(text_chunks):
                chunk_trim = chunk_content.strip()
                if not chunk_trim:
                    continue  # Skip empty chunks
                    
                logging.info(f"Analyzing chunk {chunk_index + 1}/{len(text_chunks)}")
                
                # Progress update
                progress_data = {"message": f"Analyzing section {chunk_index + 1}/{len(text_chunks)}..."}
                progress_payload = json.dumps(progress_data)
                yield f"event: progress\ndata: {progress_payload}\n\n"
                
                # Create a simpler prompt for faster inference
                prompt_local = f"""Given a document about {subject_matter} with {tone} tone, analyze:

Excerpt:
---
{chunk_trim}
---

Your brief response:"""

                # Send an initial chunk event to establish this chunk index
                chunk_data = {
                    "chunk_index": chunk_index,
                    "text_chunk": chunk_trim,
                    "analysis": "",
                    "is_complete": False
                }
                chunk_payload = json.dumps(chunk_data)
                yield f"event: chunk\ndata: {chunk_payload}\n\n"
                
                # Process the chunk and stream tokens
                try:
                    chunk_analysis_stream = run_inference(
                        prompt_local,
                        system_message=system_message_local,
                        max_new_tokens=100,  # Reduced token count for faster generation
                        stream=True
                    )
                    
                    if not chunk_analysis_stream:
                        logging.error(f"Failed to get stream for chunk {chunk_index}")
                        error_data = {
                            "chunk_index": chunk_index,
                            "error": "Failed to analyze this chunk."
                        }
                        error_payload = json.dumps(error_data)
                        yield f"event: error\ndata: {error_payload}\n\n"
                        continue
                    
                    # Initialize empty analysis for this chunk
                    current_analysis = ""
                    
                    # Stream tokens as they're generated
                    for token in chunk_analysis_stream:
                        current_analysis += token
                        
                        # Send token update
                        token_data = {
                            "chunk_index": chunk_index,
                            "token": token,
                            "is_complete": False
                        }
                        token_payload = json.dumps(token_data)
                        yield f"event: token\ndata: {token_payload}\n\n"
                
                except TimeoutError:
                    logging.error(f"Timeout during streaming for chunk {chunk_index}")
                    current_analysis += " [Analysis incomplete due to timeout]"
                except Exception as e:
                    logging.error(f"Error streaming chunk {chunk_index}: {e}")
                    current_analysis += " [Error during analysis]"
                
                # Send final completed chunk data
                complete_data = {
                    "chunk_index": chunk_index,
                    "text_chunk": chunk_trim,
                    "analysis": current_analysis,
                    "is_complete": True
                }
                complete_payload = json.dumps(complete_data)
                yield f"event: chunk\ndata: {complete_payload}\n\n"

            # Signal the end of the stream
            total_time = time.time() - start_time
            end_data = {"time_taken": f"{total_time:.2f}s"}
            end_payload = json.dumps(end_data)
            yield f"event: end\ndata: {end_payload}\n\n"
            logging.info(f"Analysis stream complete. Total time: {total_time:.2f}s")

        except Exception as e:
            logging.error(f"Error during streaming analysis: {e}", exc_info=True)
            # Yield a final error event if something unexpected happens
            error_payload = json.dumps({"error": f"An unexpected error occurred during analysis: {str(e)}"})
            yield f"event: error\ndata: {error_payload}\n\n"

    # Return the streaming response
    return Response(generate_analysis(), mimetype='text/event-stream')

if __name__ == '__main__':
    # For development, Flask's built-in server is fine
    app.run(debug=True, host='0.0.0.0', port=5001) 