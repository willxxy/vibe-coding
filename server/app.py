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
MODEL_ID = "google/gemma-3-1b-it"
model = None
tokenizer = None

def load_model_and_tokenizer():
    global model, tokenizer
    logging.info(f"Loading model: {MODEL_ID}...")
    try:
        # Removed quantization_config
        # Load model and send it to the determined device
        model = Gemma3ForCausalLM.from_pretrained(
            MODEL_ID,
            # device_map="auto" is less reliable without accelerate+bitsandbytes handling complex mapping.
            # Explicitly map to the detected device.
            # torch_dtype=torch.bfloat16 # Optional: Try bfloat16 if MPS supports it well and default precision is too slow/memory heavy
        ).to(DEVICE).eval() # Set model to evaluation mode and move to device

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        logging.info(f"Model and tokenizer loaded successfully onto {DEVICE}.")
    except Exception as e:
        logging.error(f"Error loading model: {e}")

load_model_and_tokenizer() # Load the model when the application starts
# --- End Model Loading ---

# --- Inference Helper ---
def run_inference(prompt_text, system_message="You are a helpful assistant.", max_new_tokens=128, stream=False):
    """Runs inference using the loaded model and tokenizer."""
    if not model or not tokenizer:
        logging.error("Model or tokenizer not loaded, cannot run inference.")
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

        logging.info(f"Running inference on {DEVICE} with max_new_tokens={max_new_tokens}...")
        
        if stream:
            # For streaming mode, use the streaming generator
            input_token_len = inputs["input_ids"].shape[1]
            return generate_stream(inputs, input_token_len, max_new_tokens)
        else:
            # Non-streaming mode (batch generation)
            with torch.inference_mode():
                # Generate outputs
                outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

            # Decode the output tokens
            input_token_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_token_len:]
            result_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

            logging.info("Inference completed.")
            return result_text.strip()

    except Exception as e:
        logging.error(f"Error during inference: {e}")
        return None

def generate_stream(inputs, input_token_len, max_new_tokens=128):
    """Generator function that yields tokens as they're generated."""
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

# Main analysis endpoint - Modified for Streaming
@app.route('/analyze', methods=['POST'])
def analyze_document():
    if not model or not tokenizer:
         # Still return JSON for initial errors before streaming starts
         return jsonify({"error": "Model not loaded. Check server logs."}), 500

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
            # 1. Determine Overall Tone (Global LLM task)
            logging.info("Determining overall tone...")
            tone_prompt = f"""Analyze the following text and determine its primary purpose or tone. Respond with only ONE OR TWO words from this list: [Personal Reflection, Academic Note, Creative Writing, Technical Instruction, Correction Needed, Idea Generation, Factual Summary, Question Posing].\n\n                        Text:\n                        ---\n                        {full_text}\n                        ---\n\n                        Primary Purpose/Tone:"""
            
            # First generate tone - not using streaming for this part
            determined_tone = run_inference(tone_prompt, system_message="You are an expert text analyst.", max_new_tokens=10)

            if not determined_tone:
                logging.error("Failed to determine text tone. Stopping stream.")
                # Yield an error event for the frontend
                error_payload = json.dumps({"error": "Failed to determine text tone."})
                yield f"event: error\ndata: {error_payload}\n\n"
                return # Stop the generator

            logging.info(f"Determined tone: {determined_tone}")
            # Yield the tone event
            tone_payload = json.dumps({"overall_tone": determined_tone})
            yield f"event: tone\ndata: {tone_payload}\n\n"

            # --- Analyze Chunks (Local LLM Task) ---
            logging.info(f"Splitting text into sentences...")
            sentences = split_into_sentences(full_text)
            logging.info(f"Found {len(sentences)} sentences. Chunking into groups of {chunk_size_n}...")

            chunk_index = 0
            for sentence_chunk in chunk_sentences(sentences, chunk_size_n):
                chunk_text = " ".join(sentence_chunk)
                logging.info(f"Analyzing chunk {chunk_index + 1}...")

                system_message_local = f"You are an assistant analyzing a text identified as: {determined_tone}."
                prompt_local = f"""Given the overall text is identified as '{determined_tone}', analyze the following excerpt.\n\n                        Provide a brief analysis, suggestion, or reflection relevant to the tone. For example:\n                        - If 'Correction Needed', suggest grammatical or factual corrections.\n                        - If 'Personal Reflection', offer a thoughtful comment or question.\n                        - If 'Academic Note', summarize the key point or ask a clarifying question.\n                        - If 'Idea Generation', suggest a related idea or how to expand on it.\n\n                        Excerpt:\n                        ---\n                        {chunk_text}\n                        ---\n\n                        Analysis/Suggestion:"""

                # Use streaming mode for chunk analysis
                chunk_analysis_stream = run_inference(
                    prompt_local,
                    system_message=system_message_local,
                    max_new_tokens=100,
                    stream=True
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