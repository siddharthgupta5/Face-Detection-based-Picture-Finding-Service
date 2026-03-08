"""
run.py — entry point for the PhotoFinder application.
"""

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_TELEMETRY"]     = "False"

import argparse
import logging
import uvicorn

logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the PhotoFinder server.")
    parser.add_argument("--host",   default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port",   default=8000, type=int, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true",   help="Enable auto-reload (development)")
    args = parser.parse_args()

    print(f"\n  PhotoFinder running at  http://{args.host}:{args.port}\n")

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
