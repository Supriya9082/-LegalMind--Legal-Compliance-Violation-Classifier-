"""
Script: Start the FastAPI inference server.

Usage:
  python scripts/serve.py
  python scripts/serve.py --host 0.0.0.0 --port 8000
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",   default="0.0.0.0")
    parser.add_argument("--port",   type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn
    uvicorn.run(
        "api.server:app",
        host    = args.host,
        port    = args.port,
        reload  = args.reload,
        log_level = "info",
    )


if __name__ == "__main__":
    main()
