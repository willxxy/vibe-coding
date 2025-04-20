import torch
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import AutoTokenizer, Gemma3ForCausalLM
import logging
import re
import json
import functools
import time
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

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

# --- Model Loading ---
MODEL_ID = "google/gemma-3-1b-it"  # Same model for both global and local LLM

# Global variables for models and tokenizers
model = None
tokenizer = None
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

# --- Inference Helper ---
@timeout(60)
def run_inference(prompt_text, system_message="You are a helpful assistant.", max_new_tokens=128, stream=False):
    """Runs inference using the model and tokenizer."""
    
    if not model or not tokenizer:
        logging.error("Model or tokenizer not loaded, cannot run inference.")
        return None

    try:
        # Truncate very long inputs to prevent slow processing or OOM
        max_input_length = 2048
        original_length = len(prompt_text)
        
        if len(prompt_text) > max_input_length:
            logging.warning(f"Input text truncated from {len(prompt_text)} to {max_input_length} chars")
            
            # Try to truncate at a sensible point like a paragraph break
            truncation_point = prompt_text.rfind("\n\n", 0, max_input_length)
            if truncation_point == -1 or truncation_point < max_input_length * 0.5:
                # No good paragraph break, try sentence
                truncation_point = prompt_text.rfind(". ", 0, max_input_length)
            
            if truncation_point != -1 and truncation_point > max_input_length * 0.5:
                prompt_text = prompt_text[:truncation_point + 1] + "\n\n... [text truncated]"
            else:
                # No good break point found, truncate directly
                prompt_text = prompt_text[:max_input_length] + "\n\n... [text truncated]"
        
        # Create the message format expected by Gemma 3
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

        # Apply chat template with error handling
        try:
            inputs = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(DEVICE)
        except Exception as e:
            logging.error(f"Error applying chat template: {e}")
            # Fallback to simpler prompt construction
            prompt = f"{system_message}\n\nUser: {prompt_text}\n\nAssistant:"
            inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        input_token_len = inputs["input_ids"].shape[1]
        logging.info(f"Running inference with {input_token_len} input tokens and max_new_tokens={max_new_tokens}")
        
        if input_token_len > 4096:
            logging.warning(f"Very large input: {input_token_len} tokens exceeds safe limit. Truncating.")
            # Truncate to avoid memory issues
            inputs["input_ids"] = inputs["input_ids"][:, -4096:]
            if "attention_mask" in inputs:
                inputs["attention_mask"] = inputs["attention_mask"][:, -4096:]
            input_token_len = 4096
        
        # Free up memory before generation
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
                    temperature=0.7,  # Add slight temperature for better responses
                    num_beams=1  # Simple greedy decoding is faster
                )

            # Decode the output tokens
            generated_tokens = outputs[0][input_token_len:]
            result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            elapsed = time.time() - start_time
            token_rate = len(generated_tokens) / elapsed if elapsed > 0 else 0
            logging.info(f"Inference completed in {elapsed:.2f}s ({token_rate:.1f} tokens/sec)")
            return result_text.strip()

    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA out of memory during inference")
        # Try to recover memory
        torch.cuda.empty_cache()
        return "Error: GPU memory exceeded during generation."
    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return None

@timeout(30)
def generate_stream(inputs, input_token_len, max_new_tokens=128):
    """Generator function that yields tokens as they're generated."""
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer not provided for streaming.")
        return
        
    try:
        with torch.inference_mode():
            # Track generation start time
            start_time = time.time()
            
            # Initialize generation with input
            generation = inputs["input_ids"].clone()
            
            # Keep track of tokens for timing
            tokens_generated = 0
            last_log_time = start_time
            
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
                    tokens_generated += 1
                    yield new_token_text
                
                # Log progress periodically without affecting stream
                current_time = time.time()
                if current_time - last_log_time > 2.0:  # Log every 2 seconds
                    elapsed = current_time - start_time
                    token_rate = tokens_generated / elapsed if elapsed > 0 else 0
                    logging.info(f"Generated {tokens_generated} tokens in {elapsed:.2f}s ({token_rate:.1f} tokens/sec)")
                    last_log_time = current_time
                    
                # Check for timeout
                if time.time() - start_time > 25:  # Hard cutoff before decorator timeout
                    logging.warning("Streaming generation approaching timeout, stopping early")
                    yield " [Generation stopped due to timeout]"
                    break
    
    except torch.cuda.OutOfMemoryError:
        logging.error("CUDA out of memory during token generation")
        yield " [Error: GPU memory exceeded]"
    except Exception as e:
        logging.error(f"Error during token generation: {e}")
        yield f" [Error during generation: {str(e)[:50]}...]"

# --- Text Processing Helpers ---
def split_into_sentences(text):
    """Split text into sentences using regex pattern matching."""
    if not text or not text.strip():
        return []
        
    # Clean up text - normalize spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Add space after punctuation for splitting, handle existing spaces
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    
    # Better sentence splitting with handling for common abbreviations
    # This pattern looks for sentence-ending punctuation followed by a space and capital letter
    # but excludes common abbreviations and other edge cases
    sentence_pattern = r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+(?=[A-Z])'
    sentences = re.split(sentence_pattern, text)
    
    # Further split by explicit newlines that might indicate paragraph breaks
    result = []
    for sentence in sentences:
        # Split by paragraph breaks if they exist
        paragraphs = re.split(r'\n{2,}', sentence)
        for para in paragraphs:
            if para.strip():
                # For very long paragraphs, try to break them at punctuation
                if len(para) > 300:
                    parts = re.split(r'(?<=[.!?])\s+', para)
                    result.extend([p.strip() for p in parts if p.strip()])
                else:
                    result.append(para.strip())
    
    # Filter empty strings and strip whitespace
    result = [part.strip() for part in result if part.strip()]
    
    # Handle case where splitting didn't work (e.g., no punctuation)
    if not result and text.strip():
        # For long text without punctuation, split by length
        if len(text) > 200:
            # Split into roughly equal chunks of ~150 chars
            chunk_size = 150
            chunks = [text[i:i+chunk_size].strip() for i in range(0, len(text), chunk_size)]
            result = [chunk for chunk in chunks if chunk]
        else:
            result = [text.strip()]
    
    return result

def create_text_chunks(text, chunk_size=3, overlap=1):
    """Split text into overlapping chunks of roughly n sentences each with improved handling for various text formats."""
    sentences = split_into_sentences(text)
    
    # If no sentences found, return the whole text as one chunk
    if not sentences:
        return [text] if text.strip() else []
    
    # Use a smaller chunk size for very long sentences
    avg_sentence_length = sum(len(s) for s in sentences) / len(sentences)
    if avg_sentence_length > 150:
        logging.info(f"Text has very long sentences (avg {avg_sentence_length:.1f} chars), reducing chunk size")
        chunk_size = max(2, chunk_size - 1)
    
    chunks = []
    i = 0
    
    while i < len(sentences):
        # Get chunk of size chunk_size or remaining sentences
        end_idx = min(i + chunk_size, len(sentences))
        chunk = " ".join(sentences[i:end_idx])
        
        # Try to balance chunks that are too short or too long
        if len(chunk) < 100 and end_idx < len(sentences):
            # Chunk is too small, add another sentence if available
            end_idx = min(end_idx + 1, len(sentences))
            chunk = " ".join(sentences[i:end_idx])
        elif len(chunk) > 800 and end_idx - i > 1:
            # Chunk is too large, reduce size
            end_idx = i + max(1, chunk_size - 1)
            chunk = " ".join(sentences[i:end_idx])
        
        chunks.append(chunk)
        
        # Move window with overlap
        i = i + max(1, chunk_size - overlap)
    
    return chunks

# --- API Routes ---
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
        
    chunk_size = data.get('chunk_size', 3)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        return jsonify({"error": "'chunk_size' must be a positive integer"}), 400
        
    full_text = data['text']
    if not full_text.strip():
        return jsonify({"error": "'text' cannot be empty"}), 400

    def generate_analysis():
        try:
            # Start timer for the whole process
            start_time = time.time()
            
            # Send initial progress update to client
            yield f"event: progress\ndata: {json.dumps({'message': 'Starting document analysis...'})}\n\n"
            
            # --- Global LLM Analysis ---
            # Limit text size for analysis
            text_for_analysis = full_text
            if len(text_for_analysis) > 8000:
                logging.warning(f"Document too large ({len(text_for_analysis)} chars), truncating for analysis")
                text_for_analysis = full_text[:8000] + "... [document truncated for analysis]"
            
            # Improved Global analysis prompt
            global_prompt = f"""Analyze the following document and provide three specific elements:

1. TONE: What is the emotional or stylistic quality of this text? (e.g., formal, informal, academic, personal, technical, casual, serious, humorous)
2. SUBJECT MATTER: What general category or domain does this text belong to? (e.g., journal entry, technical documentation, creative writing, academic notes)
3. CONTEXT SUMMARY: Provide a 1-2 sentence summary capturing the key context of this document.

TEXT TO ANALYZE:
----------------
{text_for_analysis}
----------------

Your response MUST follow this exact format:
TONE: [single word for tone]
SUBJECT: [1-2 words for subject matter]
SUMMARY: [1-2 sentence summary]
"""
            
            # Progress update
            yield f"event: progress\ndata: {json.dumps({'message': 'Analyzing document context...'})}\n\n"
            
            # Run global analysis
            global_analysis_text = run_inference(
                global_prompt, 
                system_message="You are an expert text analyst. Provide precise, structured analysis following the exact format requested.", 
                max_new_tokens=150
            )

            if not global_analysis_text:
                logging.error("Failed to complete global analysis. Stopping stream.")
                yield f"event: error\ndata: {json.dumps({'error': 'Failed to determine global text attributes.'})}\n\n"
                return
            
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
            yield f"event: global_analysis\ndata: {json.dumps(global_analysis)}\n\n"
            
            # --- Local LLM Processing ---
            # Progress update
            yield f"event: progress\ndata: {json.dumps({'message': 'Preparing text chunks...'})}\n\n"
            
            # Create text chunks with minimal overlap for better coherence
            text_chunks = create_text_chunks(full_text, chunk_size=min(chunk_size, 4), overlap=1)
            
            # Limit number of chunks for performance
            max_chunks = 15
            if len(text_chunks) > max_chunks:
                logging.warning(f"Too many chunks ({len(text_chunks)}), limiting to {max_chunks}")
                text_chunks = text_chunks[:max_chunks]
            
            # Handle empty text case
            if not text_chunks:
                text_chunks = [""]
                
            logging.info(f"Created {len(text_chunks)} chunks for analysis")

            # Improved system message for local LLM with clearer context
            system_message_local = f"""You are an AI assistant analyzing document excerpts.
Context information: This document has been identified as having a {tone} tone and is about {subject_matter}.
Overall document summary: {context_summary}

Your task is to provide insightful, helpful analysis for each excerpt based on this context.
Keep responses concise, focused, and relevant to both the specific excerpt and the overall document context."""

            # Process chunks
            for chunk_index, chunk_content in enumerate(text_chunks):
                chunk_trim = chunk_content.strip()
                if not chunk_trim:
                    continue  # Skip empty chunks
                    
                logging.info(f"Analyzing chunk {chunk_index + 1}/{len(text_chunks)}")
                
                # Progress update
                yield f"event: progress\ndata: {json.dumps({'message': f'Analyzing section {chunk_index + 1}/{len(text_chunks)}...'})}\n\n"
                
                # Improved prompt for local LLM with better context integration
                prompt_local = f"""Analyze the following excerpt from a {subject_matter} document with {tone} tone:

EXCERPT:
--------
{chunk_trim}
--------

Provide a brief, focused response that:
1. Addresses this specific excerpt directly
2. Maintains consistency with the document's overall context
3. Offers valuable insights appropriate to this type of content
"""

                # Send chunk initialization event
                chunk_data = {
                    "chunk_index": chunk_index,
                    "text_chunk": chunk_trim,
                    "analysis": "",
                    "is_complete": False
                }
                yield f"event: chunk\ndata: {json.dumps(chunk_data)}\n\n"
                
                # Process the chunk and stream tokens
                try:
                    chunk_analysis_stream = run_inference(
                        prompt_local,
                        system_message=system_message_local,
                        max_new_tokens=150,
                        stream=True
                    )
                    
                    if not chunk_analysis_stream:
                        logging.error(f"Failed to get stream for chunk {chunk_index}")
                        yield f"event: error\ndata: {json.dumps({'chunk_index': chunk_index, 'error': 'Failed to analyze this chunk.'})}\n\n"
                        continue
                    
                    # Initialize analysis text
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
                        yield f"event: token\ndata: {json.dumps(token_data)}\n\n"
                
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
                yield f"event: chunk\ndata: {json.dumps(complete_data)}\n\n"

            # Signal the end of the stream
            total_time = time.time() - start_time
            yield f"event: end\ndata: {json.dumps({'time_taken': f'{total_time:.2f}s'})}\n\n"
            logging.info(f"Analysis stream complete. Total time: {total_time:.2f}s")

        except Exception as e:
            logging.error(f"Error during streaming analysis: {e}", exc_info=True)
            yield f"event: error\ndata: {json.dumps({'error': f'An unexpected error occurred during analysis: {str(e)}'})}\n\n"

    # Return the streaming response
    return Response(generate_analysis(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 