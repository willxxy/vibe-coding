#!/usr/bin/env python3
"""
Production server launcher using Gunicorn.
This script provides a convenient way to start the server in production mode.
"""

import os
import sys
import argparse
import logging
import subprocess
import time
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("server_launcher")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Launch the LLM server in production mode.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", default=5001, type=int, help="Port to bind to (default: 5001)")
    parser.add_argument("--workers", default=1, type=int, 
                        help="Number of worker processes (default: 1, more workers use more memory)")
    parser.add_argument("--debug", action="store_true", help="Run in debug mode using Flask's development server")
    parser.add_argument("--timeout", default=120, type=int, help="Worker timeout in seconds (default: 120)")
    parser.add_argument("--preload", action="store_true", 
                        help="Preload model before starting workers (good for production, uses more memory)")
    parser.add_argument("--reload", action="store_true",
                        help="Auto-reload server when code changes (development only)")
    return parser.parse_args()

def preload_model():
    """Preload the model to warm up cache before starting the server."""
    logger.info("Preloading model to warm up cache...")
    start_time = time.time()
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        # Import model loading function and run it
        from app import load_model_and_tokenizer
        success = load_model_and_tokenizer()
        
        if success:
            elapsed = time.time() - start_time
            logger.info(f"Model preloaded successfully in {elapsed:.2f}s")
            return True
        else:
            logger.error("Failed to preload model")
            return False
    except Exception as e:
        logger.error(f"Error during model preloading: {e}")
        return False

def preload_async():
    """Preload model in a separate thread to avoid blocking server startup."""
    def _preload_worker():
        try:
            preload_model()
        except Exception as e:
            logger.error(f"Async preload failed: {e}")
    
    thread = threading.Thread(target=_preload_worker)
    thread.daemon = True
    thread.start()
    logger.info("Started async model preloading")
    return thread

def main():
    """Main function to launch the server."""
    args = parse_args()
    
    # Set environment variables
    os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is not buffered
    
    # Get the absolute path of the app.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Preload model if requested
    preload_thread = None
    if args.preload and not args.debug:
        if args.workers > 1:
            # For multiple workers, preload synchronously to avoid duplicate loading
            success = preload_model()
            if not success:
                logger.warning("Continuing without preloaded model. It will be loaded on first request.")
        else:
            # For single worker, we can preload asynchronously
            preload_thread = preload_async()
    
    if args.debug:
        logger.info(f"Starting Flask development server on {args.host}:{args.port} in debug mode")
        
        # Import and run Flask app directly in debug mode
        sys.path.insert(0, current_dir)
        from app import app as flask_app
        flask_app.run(host=args.host, port=args.port, debug=True)
    else:
        logger.info(f"Starting Gunicorn server on {args.host}:{args.port} with {args.workers} workers")
        
        # Construct Gunicorn command
        cmd = [
            "gunicorn",
            "--bind", f"{args.host}:{args.port}",
            "--workers", str(args.workers),
            "--timeout", str(args.timeout),
            "--worker-class", "sync",  # Use sync workers for Flask
            "--log-level", "info"
        ]
        
        # Add preload flag if requested
        if args.preload:
            cmd.append("--preload")
            
        # Add reload flag if requested
        if args.reload:
            cmd.append("--reload")
            
        # Add app path
        cmd.append("app:app")  # app.py main Flask app
        
        try:
            # Wait for preload thread to finish if it exists
            if preload_thread and preload_thread.is_alive():
                logger.info("Waiting for model preloading to complete...")
                preload_thread.join(timeout=60)  # Don't wait forever
                
            subprocess.run(cmd, cwd=current_dir, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Gunicorn server: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
            sys.exit(0)

if __name__ == "__main__":
    main() 