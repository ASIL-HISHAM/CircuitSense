#!/usr/bin/env python3
"""
CircuitSense — API Server Startup
Run FastAPI server with uvicorn
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import uvicorn
from src.api import app
from src.core import get_config, setup_logging

def main():
    """Start the FastAPI server."""
    # Load configuration
    config = get_config()
    
    # Setup logging
    setup_logging(
        level=config.api.log_level,
        module_name="circuitsense.api"
    )
    
    # Start server
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║          CircuitSense API Server v2.0.0                      ║
║          Neural fault diagnosis from circuit waveforms       ║
╚══════════════════════════════════════════════════════════════╝

Starting server at http://{config.api.host}:{config.api.port}
📚 API Documentation: http://{config.api.host}:{config.api.port}/docs
🧪 Alternative UI: http://{config.api.host}:{config.api.port}/redoc

""")
    
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level=config.api.log_level.lower(),
    )


if __name__ == "__main__":
    main()
