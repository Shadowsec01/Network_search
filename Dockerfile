# =============================================================================
# Dockerfile — Communicating Agents Network Search AI
# =============================================================================
#
# WHAT IS A DOCKERFILE?
#   A script that builds a Docker "image" — a portable, self-contained
#   environment with Python, your code, and all dependencies bundled in.
#   This image runs identically on Windows, Mac, Linux, or any cloud server.
#
# BUILD STAGES EXPLAINED:
#   FROM    → Base operating system layer (Python 3.13 on slim Debian)
#   WORKDIR → Sets /app as the working directory inside the container
#   COPY    → Copies files from your PC into the container image
#   RUN     → Executes shell commands during image build (installs packages)
#   EXPOSE  → Documents which port the app listens on (informational)
#   CMD     → Command that runs when the container starts
#
# =============================================================================

# ── Base image: official Python 3.13 slim (Debian-based, minimal size ~50MB)
FROM python:3.13-slim

# ── Set the working directory inside the container
WORKDIR /app

# ── Copy dependency list first (Docker cache optimisation)
#    If requirements.txt hasn't changed, Docker reuses the cached pip layer
#    and skips reinstalling packages — speeds up rebuilds significantly.
COPY requirements.txt .

# ── Install Python dependencies
#    --no-cache-dir : Don't store pip cache in image (keeps image smaller)
#    --upgrade      : Ensure latest compatible versions
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# ── Copy the rest of the project into the container
COPY . .

# ── Expose port 5000 (Flask default)
#    This is metadata only — actual port mapping is in docker-compose.yml
EXPOSE 5000

# ── Start command: Gunicorn WSGI server
#    Gunicorn is a production-grade server that replaces Flask's built-in server.
#    Flask's built-in server is single-threaded and not suitable for production.
#
#    Arguments explained:
#      -w 2          : 2 worker processes (handles 2 concurrent requests)
#      -b 0.0.0.0:5000 : Bind to all interfaces on port 5000
#      --timeout 120 : Kill workers that hang for >120 seconds
#      app:app       : Python module "app", Flask instance named "app"
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
