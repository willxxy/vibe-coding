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

def main():
    """Main function to launch the server."""
    args = parse_args()
    
    # Set environment variables
    os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is not buffered
    
    # Get the absolute path of the app.py file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(current_dir, "app.py")
    
    # Preload model if requested
    if args.preload and not args.debug:
        success = preload_model()
        if not success:
            logger.warning("Continuing without preloaded model. It will be loaded on first request.")
    
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
            
        # Add app path
        cmd.append("app:app")  # app.py main Flask app
        
        try:
            subprocess.run(cmd, cwd=current_dir, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to start Gunicorn server: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            logger.info("Server stopped by user")
            sys.exit(0)

if __name__ == "__main__":
    main() 