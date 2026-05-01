"""
run.py — Production Server Launcher

Development (auto-reload, single worker):
    python run.py dev

Production (multiple workers, no reload):
    python run.py prod
"""

import sys
import uvicorn

MODE = sys.argv[1] if len(sys.argv) > 1 else "dev"

if MODE == "prod":
    # Production: multiple workers to handle concurrent users.
    # Use 1 worker per CPU core, e.g. 4 cores = 4 workers.
    # NOTE: On Windows, uvicorn multiprocessing requires the 'spawn' start method.
    # For true multi-worker production on Windows, use a cloud Linux container instead.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",   # Expose to all network interfaces (required in Docker)
        port=8000,
        workers=1,         # Keep at 1 on Windows; scale via Docker replicas in cloud
        reload=False,      # NEVER use reload in production — it kills performance
        log_level="info",
        access_log=True,
    )
else:
    # Development: single worker with auto-reload on file save
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="debug",
    )
