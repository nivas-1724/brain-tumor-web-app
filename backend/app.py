"""
Brain Tumor Detection — Backend Entrypoint Proxy
Ensures both `gunicorn app:app` and `gunicorn backend.app:app` work seamlessly on Render.
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app
