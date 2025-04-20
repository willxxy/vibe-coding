import torch
from flask import Flask, request, jsonify, Response # Import Response
from flask_cors import CORS # Import CORS
from transformers import AutoTokenizer, Gemma3ForCausalLM
import logging
import re # For simple sentence splitting
import json # Import json for formatting SSE data

# Configure logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
CORS(app) # Enable CORS for the entire app

# --- Device Setup ---
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    # elif torch.cuda.is_available(): # Keep commented out unless needed later
    #     return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_device()
logging.info(f"Using device: {DEVICE}")
# --- End Device Setup ---

# --- Model Loading ---
GLOBAL_MODEL_ID = "google/gemma-3-1b-it"  # Model for document-wide analysis
LOCAL_MODEL_ID = "google/gemma-3-1b-it"   # Model for chunk processing

global_model = None
global_tokenizer = None
local_model = None
local_tokenizer = None

def load_models_and_tokenizers():
    global global_model, global_tokenizer, local_model, local_tokenizer
    
    # Load Global LLM
    logging.info(f"Loading global model: {GLOBAL_MODEL_ID}...")
    try:
        global_model = Gemma3ForCausalLM.from_pretrained(
            GLOBAL_MODEL_ID,
        ).to(DEVICE).eval()
        global_tokenizer = AutoTokenizer.from_pretrained(GLOBAL_MODEL_ID)
        logging.info(f"Global model loaded successfully onto {DEVICE}.")
    except Exception as e:
        logging.error(f"Error loading global model: {e}")
    
    # Load Local LLM
    logging.info(f"Loading local model: {LOCAL_MODEL_ID}...")
    try:
        local_model = Gemma3ForCausalLM.from_pretrained(
            LOCAL_MODEL_ID,
        ).to(DEVICE).eval()
        local_tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_ID)
        logging.info(f"Local model loaded successfully onto {DEVICE}.")
    except Exception as e:
        logging.error(f"Error loading local model: {e}")

load_models_and_tokenizers() # Load both models when the application starts
# --- End Model Loading ---

# --- Inference Helper ---
def run_inference(prompt_text, system_message="You are a helpful assistant.", max_new_tokens=128, stream=False, model_type="global"):
    """Runs inference using the specified model and tokenizer."""
    
    # Select the appropriate model and tokenizer
    if model_type == "global":
        model = global_model
        tokenizer = global_tokenizer
    else:  # model_type == "local"
        model = local_model
        tokenizer = local_tokenizer
    
    if not model or not tokenizer:
        logging.error(f"{model_type.capitalize()} model or tokenizer not loaded, cannot run inference.")
        return None

    messages = [
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_message}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            },
        ],
    ]

    try:
        # Use the tokenizer's chat template and move inputs to the correct device
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(DEVICE) # Move inputs to the same device as the model

        logging.info(f"Running {model_type} inference on {DEVICE} with max_new_tokens={max_new_tokens}...")
        
        if stream:
            # For streaming mode, use the streaming generator
            input_token_len = inputs["input_ids"].shape[1]
            return generate_stream(inputs, input_token_len, max_new_tokens, model, tokenizer)
        else:
            # Non-streaming mode (batch generation)
            with torch.inference_mode():
                # Generate outputs
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            # Decode the output tokens
            input_token_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_token_len:]
            result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            logging.info(f"{model_type.capitalize()} inference completed.")
            return result_text.strip()

    except Exception as e:
        logging.error(f"Error during {model_type} inference: {e}")
        return None

def generate_stream(inputs, input_token_len, max_new_tokens=128, model=None, tokenizer=None):
    """Generator function that yields tokens as they're generated."""
    # Use the provided model and tokenizer
    if model is None or tokenizer is None:
        logging.error("Model or tokenizer not provided for streaming.")
        return
        
    generated_text = ""
    
    # Initialize generation with input ids 
    generation = inputs["input_ids"].clone()
    
    with torch.inference_mode():
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
                generated_text += new_token_text
                yield new_token_text

# --- End Inference Helper ---

# --- Text Processing Helpers ---
def split_into_sentences(text):
    """Rudimentary sentence splitting based on punctuation."""
    # Add space after punctuation for splitting, handle existing spaces
    text = re.sub(r'([.!?])\s*', r'\1 ', text)
    sentences = [s.strip() for s in text.split('  ') if s.strip()] # Split by double space, filter empty
    # A better approach would use NLTK or spaCy
    # Example: nltk.sent_tokenize(text)
    processed_sentences = []
    for sentence in sentences:
        # Simple regex split - might imperfectly handle abbreviations etc.
        parts = re.split(r'(?<=[.!?])\s+', sentence)
        processed_sentences.extend([p.strip() for p in parts if p.strip()])
    return processed_sentences

def chunk_sentences(sentences, n):
    """Yield successive n-sized chunks from sentences."""
    for i in range(0, len(sentences), n):
        yield sentences[i:i + n]

# --- End Text Processing Helpers ---


@app.route('/')
def home():
    return "Backend server is running!"

# Main analysis endpoint - Modified for two-LLM approach
@app.route('/analyze', methods=['POST'])
def analyze_document():
    if not global_model or not global_tokenizer or not local_model or not local_tokenizer:
         # Still return JSON for initial errors before streaming starts
         return jsonify({"error": "Models not loaded. Check server logs."}), 500

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
            logging.info("Performing global document analysis...")
            
            # 1. Determine Overall Tone
            tone_prompt = f"""Analyze the following text and determine its primary purpose or tone. Respond with only ONE OR TWO words from this list: [Personal Reflection, Academic Note, Creative Writing, Technical Instruction, Correction Needed, Idea Generation, Factual Summary, Question Posing].\n\n                        Text:\n                        ---\n                        {full_text}\n                        ---\n\n                        Primary Purpose/Tone:"""
            
            determined_tone = run_inference(
                tone_prompt, 
                system_message="You are an expert text analyst.", 
                max_new_tokens=10, 
                model_type="global"
            )

            if not determined_tone:
                logging.error("Failed to determine text tone. Stopping stream.")
                error_payload = json.dumps({"error": "Failed to determine text tone."})
                yield f"event: error\ndata: {error_payload}\n\n"
                return
            
            logging.info(f"Determined tone: {determined_tone}")
            
            # 2. Determine Subject Matter
            subject_prompt = f"""Analyze the following text and determine its subject matter or domain. Respond with only ONE OR TWO words describing what this text is about.\n\n                        Text:\n                        ---\n                        {full_text}\n                        ---\n\n                        Subject Matter:"""
            
            subject_matter = run_inference(
                subject_prompt, 
                system_message="You are an expert text analyst.", 
                max_new_tokens=10, 
                model_type="global"
            )
            
            logging.info(f"Determined subject matter: {subject_matter}")
            
            # 3. Generate a brief context summary
            summary_prompt = f"""Provide a brief summary (2-3 sentences) of the following text that captures its main points or context.\n\n                        Text:\n                        ---\n                        {full_text}\n                        ---\n\n                        Summary:"""
            
            context_summary = run_inference(
                summary_prompt, 
                system_message="You are an expert text analyst.", 
                max_new_tokens=100, 
                model_type="global"
            )
            
            logging.info(f"Generated context summary")
            
            # Combine global analysis results
            global_analysis = {
                "tone": determined_tone,
                "subject_matter": subject_matter,
                "context_summary": context_summary
            }
            
            # Send global analysis to client
            global_payload = json.dumps(global_analysis)
            yield f"event: global_analysis\ndata: {global_payload}\n\n"
            
            # --- Local LLM Processing (Chunk-based) ---
            logging.info(f"Splitting text into sentences for local processing...")
            sentences = split_into_sentences(full_text)
            logging.info(f"Found {len(sentences)} sentences. Chunking into groups of {chunk_size_n}...")

            chunk_index = 0
            for sentence_chunk in chunk_sentences(sentences, chunk_size_n):
                chunk_text = " ".join(sentence_chunk)
                logging.info(f"Analyzing chunk {chunk_index + 1} with local LLM...")

                # Create a system message that includes the global context
                system_message_local = f"""You are analyzing text that has been identified as:
                - Tone: {determined_tone}
                - Subject: {subject_matter}
                - Context: {context_summary}
                
                Provide responses appropriate for this context."""
                
                prompt_local = f"""Given the overall document context, analyze the following excerpt and provide a helpful response. Only generate the response and nothing else.
                
                Excerpt:
                ---
                {chunk_text}
                ---
                
                Your analysis/response:"""

                # Use local LLM for streaming chunk analysis
                chunk_analysis_stream = run_inference(
                    prompt_local,
                    system_message=system_message_local,
                    max_new_tokens=100,
                    stream=True,
                    model_type="local"
                )
                
                # Initialize empty analysis for this chunk
                current_analysis = ""
                
                # Send an initial chunk event to establish this chunk index
                result_data = {
                    "chunk_index": chunk_index,
                    "text_chunk": chunk_text,
                    "analysis": "",
                    "is_complete": False
                }
                chunk_payload = json.dumps(result_data)
                yield f"event: chunk\ndata: {chunk_payload}\n\n"
                
                # Stream tokens as they're generated
                try:
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
                except Exception as e:
                    logging.error(f"Error streaming chunk {chunk_index}: {e}")
                    current_analysis = "[Error during analysis]"
                
                # Send final completed chunk data
                complete_data = {
                    "chunk_index": chunk_index,
                    "text_chunk": chunk_text,
                    "analysis": current_analysis,
                    "is_complete": True
                }
                complete_payload = json.dumps(complete_data)
                yield f"event: chunk\ndata: {complete_payload}\n\n"
                
                chunk_index += 1

            # Signal the end of the stream
            yield f"event: end\ndata: Stream finished\n\n"
            logging.info("Analysis stream complete.")

        except Exception as e:
            logging.error(f"Error during streaming analysis: {e}", exc_info=True)
            # Yield a final error event if something unexpected happens
            error_payload = json.dumps({"error": f"An unexpected error occurred during analysis: {str(e)}"})
            yield f"event: error\ndata: {error_payload}\n\n"

    # Return the streaming response
    return Response(generate_analysis(), mimetype='text/event-stream')

if __name__ == '__main__':
    # Consider using a production-ready server like Gunicorn or Waitress
    # For development, Flask's built-in server is fine.
    # Ensure host='0.0.0.0' if you want to access it from other devices on your network
    app.run(debug=True, host='0.0.0.0', port=5001) # Running on a different port than the default 5000 